import os
import numpy as np
#import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Tuple
#from tensorflow.data import Dataset
#from tensorflow.keras.preprocessing import image
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,SequentialSampler
#import torchvision.models as models
#img_model=models.resnet152()
from torchvision.models import resnet152,resnet
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPConfig, CLIPModel, CLIPVisionModel

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 768
        self.image_size = config.image_size  # 224
        self.patch_size = config.patch_size  # 32

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        n_patch = self.num_patches
        self.pixel_position_embedding = nn.Embedding(n_patch, self.embed_dim)
        self.register_buffer("pixel_position_ids", torch.arange(n_patch).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds

        pixel_values=self.patch_embedding(pixel_values)
        pixel_values=pixel_values.flatten(2).transpose(1,2)
        pixel_values=pixel_values+self.pixel_position_embedding(self.pixel_position_ids)

        embeddings=torch.cat((embeddings,pixel_values),dim=1)

        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*16, 768 3个子图
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds)  # bsz, 48, 768
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)

        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*4, 768 3个子图
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds)  # bsz, 12, 768
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings

class VisionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class HfClipViTImageModel(nn.Module):
    """Hugging Face CLIP 视觉塔（带预训练权重）：适合 ViT-L/14 等升级。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = CLIPVisionModel.from_pretrained(self.args.vit_name)
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        h = out.last_hidden_state
        x_global = h[:, 0, :]
        return x_global, h


class ViTImageModel(nn.Module):
    """
    视觉编码器：始终使用 HF CLIPVisionModel.from_pretrained，加载完整预训练 CLIP 视觉塔。

    返回：
        x_global : [B, hidden]            CLS token 表征
        h        : [B, num_patches+1, hidden]   含 CLS 的全部 token 表征
    """

    def __init__(self, args):
        super(ViTImageModel, self).__init__()
        self.args = args

        # 始终走 HF 视觉塔；保留 _hf 字段以便外部代码兼容判定
        self._hf = HfClipViTImageModel(args)
        self.vision_config = self._hf.encoder.config
        # 下面这些字段在新前向路径里不再使用，保留为 None 避免 state_dict 误命中旧权重
        self.vision_embeddings = None
        self.vision_pre_layrnorm = None
        self.vision_post_layernorm = None
        self.vision_pooler = None

        # 若 args.use_hf_clip_vision 显式传了 False，提示一下（但仍强制走 HF）
        if not getattr(args, "use_hf_clip_vision", True):
            import warnings as _w

            _w.warn(
                "[ViTImageModel] args.use_hf_clip_vision=False 已被忽略：旧的非 HF 路径"
                " 不加载 CLIP 预训练权重也不跑 transformer 层，已废弃。"
                "强制使用 HfClipViTImageModel。"
            )

    def forward(self, x):
        # 一致返回 (global, all_tokens)
        return self._hf(x)


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load("./resnet/resnet152-b121ed2d.pth"))
        # self.resnet=resnet152(pretrained=True)
        self.resnet = net

    def forward(self, x, att_size=7):
        # x shape batch_size * channels * 224 * 224
        # 64 * 3 * 224 * 224

        # batch_size * channels * 112 * 112
        # 64 * 64 * 112 * 112
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # 64 * 256 * 56 * 56
        x = self.resnet.maxpool(x)

        # 32 * 512 * 56 * 56
        x = self.resnet.layer1(x)
        # 32 * 512 * 28 * 28
        x = self.resnet.layer2(x)
        # 32 * 1024 * 14 * 14
        x = self.resnet.layer3(x)
        # 32 * 2048 * 7 * 7
        x = self.resnet.layer4(x)

        # 32 * 2048
        # fc = x.mean(3).mean(2)
        # 32 * 2048 * 7 * 7
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])

        # 32 * 2048 * 1 * 1
        x = self.resnet.avgpool(x)
        # 32 * 2048 * 2048
        x = x.view(x.size(0), -1)

        return x, att
def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image
def read_image (path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # args.crop_size, by default it is set to be 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    file_names = os.listdir(path)
    imgids, imgs = [], []
    dict_features=[]
    prom_num=0
    for file_name in file_names:
        image_path=os.path.join(path, file_name)
        try:
            image = image_process(image_path, transform)
        except:
            print(image_path + " has problem!")
            prom_num+=1
            image_path_fail = os.path.join(path, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)
        # print(type(image))
        # image=image.unsqueeze(0)

        imgs.append(image)
        imgids.append(file_name.split(".")[0])
       # print(imgs.shape)
    print("total num of problem images:", prom_num)
    # imgs = np.vstack(imgs)
    return (imgids, imgs)



class ImageDataset(Dataset):
    def __init__(self,examples):
        self.examples=examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


'''

class ImageLoader(object):
    def __init__(self, path: str) -> None:
        self.__feature_path = f"{path}/dict_feature.pkl"
        self.img_model =ImageModel()
        # imgs = read_image(path)
        # # print(len(imgs[0]))
        # # print(imgs[1].shape)
        # features = self.__parse_data(imgs, self.__feature_path)

        if not os.path.exists(self.__feature_path):
            imgs = read_image(path)
            features = self.__parse_data(imgs, self.__feature_path)
        else:
            features = read_pickle(self.__feature_path)
        self.__dict_features = features
        #print(len(list(self.__dict_features.values())))

        self.__feature_shape = list(self.__dict_features.values())[0].shape
        #print(self.__feature_shape)
        #exit()

    def getFeature(self, imgid: str):
        return self.__dict_features.get(imgid, None).reshape(
            (1, self.__feature_shape[-1])
        )

    def __parse_data(self, data: Tuple, save_path: str) -> Dict:
        imgids, imgs = data
        #print(imgs[0])
        #print(type(imgs))
        img_dataset=ImageDataset(imgs)
        img_sampler=SequentialSampler(img_dataset)
        img_dataloader=DataLoader(img_dataset,shuffle=False,sampler=img_sampler,batch_size=64)
        features = []
        for batch_data in tqdm(img_dataloader,ncols=80,ascii=True):
            img_global_feature,img_local_feature=self.img_model(batch_data)
            # print(type(img_global_feature))
            # print(img_global_feature.shape)
            # print(img_local_feature.shape)
            # print(img_local_feature)
            # exit()
            features.append(img_global_feature)

        features = np.vstack(features)
        # print(features.shape)
        # print(type(features))
        # exit()
        dict_features = {}
        for imgid, feature in zip(imgids, features):
            dict_features[imgid] = feature
        pd.to_pickle(dict_features, save_path)
        return dict_features

'''
if __name__ == "__main__":
    outputs=read_image("./data/twitter2017_images")
    #img_model=ImageModel()
    imgids,images=outputs
    net=getattr(resnet,'resnet152')()
    net.load_state_dict(torch.load("./resnet/resnet152-b121ed2d.pth"))
    img_model=ImageModel(net)
    #imgids, imgs = data
    # print(imgs[0])
    # print(type(imgs))
    img_dataset = ImageDataset(images)
    img_sampler = SequentialSampler(img_dataset)
    img_dataloader = DataLoader(img_dataset, shuffle=False, sampler=img_sampler, batch_size=32)
    features = []
    for batch_data in tqdm(img_dataloader, ncols=80, ascii=True):
        img_global_feature, img_local_feature = img_model(batch_data)
        print(img_local_feature.shape)
        print(img_global_feature.shape)




    # imgLoader = ImageLoader("./data/twitter2017_images")
    # feature=imgLoader.getFeature("9683")
    # print(type(feature))
    # print(feature.shape)
    # print(feature)
