import numpy as np
import torch
import random
import os
import logging
import argparse



from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True


# default params
def arg_parse():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--baseline", action="store_true",
    #                     help="Whether to run training.")

    # True for uncased BERT
    parser.add_argument(
        "--do_lower_case",
        default=True,
        # action="store_true",
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: [bert, roberta]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="pretrain/roberta-large",
        type=str,
        help="Path to pre-trained model or Hugging Face model id (e.g. roberta-large)",
    )

    parser.add_argument(
        "--config_name", default="", type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="pretrain/roberta-large",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, "
             "sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_span_len",
        default=5,
        type=int,
        help="The maximum word num of a span.",
    )

    parser.add_argument("--bert_lr", default=1e-5, type=float,
                        help="The initial learning rate for BERT.")
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The initial learning rate of classifier.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.01, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", default=1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=float, default=1.0,
                        help="Log every X updates steps.")
    parser.add_argument("--txt_global_dim", default=1024, type=int,
                        help="the dims of text sentence global embeddings.")
    parser.add_argument("--img_global_dim", default=768, type=int,
                        help="the dims of Image global embeddings.")
    parser.add_argument("--txt_local_dim", default=1024, type=int,
                        help="the dims of entity spans embeddings.")
    parser.add_argument("--img_local_dim", default=768, type=int,
                        help="the dims of Image patch embeddings.")
    parser.add_argument("--hidden_dim", default=1024, type=int,
                        help="the dims of hidden embeddings.")
    parser.add_argument(
        "--ib_kl_weight",
        default=0.001,
        type=float,
        help="Weight on MEC/VIB KL term (loss_ib_tv). ICME paper optimal β=0.1 for MEC; "
        "use 0.1 for paper-align runs, keep 0.001 for legacy roberta-large runs.",
    )
    parser.add_argument(
        "--ot_prompt_ratio",
        default=0.3,
        type=float,
        help="CAC filter-bucket ratio p in OptimalTransportFusion.add_prompt_bucket (paper p=0.3).",
    )
    parser.add_argument(
        "--ot_fusion_temperature",
        default=0.1,
        type=float,
        help="Temperature on transport_plan softmax in CAC fusion (paper λ=0.5 mapped here for repro; default 0.1).",
    )

    #======For Roberta_PGIM==================
    # parser.add_argument("--txt_global_dim", default=1024, type=int,
    #                     help="the dims of text sentence global embeddings.")
    # parser.add_argument("--img_global_dim", default=1024, type=int,
    #                     help="the dims of Image global embeddings.")
    # parser.add_argument("--txt_local_dim", default=1024, type=int,
    #                     help="the dims of entity spans embeddings.")
    # parser.add_argument("--img_local_dim", default=1024, type=int,
    #                     help="the dims of Image patch embeddings.")
    # parser.add_argument("--hidden_dim", default=768, type=int,
    #                     help="the dims of hidden embeddings.")

    return parser


def prepare_optimizer_scheduler_fusion(args, model, training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight","LayerNorm.bias"]
    # print("B:",model.parameters)
    # print("A:",args.model_type)
    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()

    other_parameters = list(model.img2uni.named_parameters()) \
                       +list(model.txt2uni.named_parameters())\
                       + list(model.span_layer.named_parameters()) \
                       + list(model.span_classifier_img.named_parameters()) \
                        +list(model.txt2img_attention.named_parameters()) \
                        +list(model.mu_img.named_parameters()) \
                        + list(model.logvar_img.named_parameters())


    #freeze resnet
    for name,param in model.named_parameters():
        if "image_model" in name:
            param.required_grad=False

    #参考：https://blog.csdn.net/winter2121/article/details/116235872
    optimizer_grouped_parameters = [
        # other params
        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr},

        # bert params
        {"params": [p for n, p in bert_parameters if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},

        {"params": [p for n, p in bert_parameters if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )

    return optimizer, scheduler

def prepare_optimizer_scheduler_MIB(args, model, training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight","LayerNorm.bias"]

    # print("B:", model.parameters)
    # print("A:", args.model_type)

    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()



    other_parameters = list(model.img2uni.named_parameters()) \
                       +list(model.txt2uni.named_parameters())\
                       + list(model.span_layer.named_parameters()) \
                       + list(model.span_classifier.named_parameters()) \
                        +list(model.bottleneck_fusion.named_parameters()) \
                       +list(model.dist_t.named_parameters()) \
                       + list(model.dist_v.named_parameters()) \
                       + list(model.tv_ib.named_parameters())\
 \
 \
    #freeze resnet
    for name,param in model.named_parameters():
        if "image_model" in name:
            param.required_grad=False

    #参考：https://blog.csdn.net/winter2121/article/details/116235872
    optimizer_grouped_parameters = [
        # other params
        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr},

        # bert params
        {"params": [p for n, p in bert_parameters if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},

        {"params": [p for n, p in bert_parameters if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )

    return optimizer, scheduler


#解释prepare_optimizer_scheduler_OT
#=================================新改进的优化器代码开始===========================================
def prepare_optimizer_scheduler_OT(args, model, training_steps):
    """
    为多模态信息瓶颈模型准备优化器和学习率调度器

    Args:
        args: 训练参数配置
        model: 多模态模型
        training_steps: 总训练步数

    Returns:
        optimizer: AdamW优化器
        scheduler: 线性预热衰减调度器
    """

    # 1. 首先冻结预训练模型参数（在创建优化器之前）
    if args.freeze_pretrained:
        freeze_pretrained_params(model)

    print("Parameters:",model.named_parameters)

    # 2. 定义不需要权重衰减的参数类型
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # 3. 安全地获取BERT参数
    bert_parameters = get_bert_parameters(model, args.model_type)

    # 4. 获取其他组件参数
    other_parameters = get_other_component_parameters(model)

    # 5. 创建优化器参数组（只包含需要训练的参数）
    optimizer_grouped_parameters = create_parameter_groups(
        bert_parameters, other_parameters, no_decay, args
    )

    # 6. 创建优化器和调度器
    optimizer = AdamW(
        optimizer_grouped_parameters,
        eps=args.adam_epsilon
    )

    warmup_steps=args.warmup_ratio * training_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps
    )
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=training_steps*0.9,  # 在训练的前90%步数内进行余弦退火
    #     eta_min=args.lr * 0.1  # 最小学习率为初始学习率的10%
    # )

    return optimizer, scheduler


def freeze_pretrained_params(model):
    """冻结预训练模型参数"""
    frozen_count = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        # 可以根据需要扩展冻结条件
        if should_freeze_param(name):
            param.requires_grad = False
            frozen_count += 1

    print(f"冻结了 {frozen_count}/{total_params} 个参数")


def should_freeze_param(param_name):
    """判断参数是否应该被冻结"""
    freeze_patterns = [
        "image_model",  # ResNet等图像编码器
        #"visual_encoder",  # 其他可能的视觉编码器命名
        # 可以根据需要添加更多模式
    ]

    return any(pattern in param_name for pattern in freeze_patterns)


def get_bert_parameters(model, model_type):
    """安全地获取BERT参数"""
    # 避免使用eval()，使用getattr更安全
    try:
        bert_module = getattr(model, model_type)
        return list(bert_module.named_parameters())
    except AttributeError:
        # 如果找不到指定的模型类型，尝试常见的命名
        common_names = ['bert', 'roberta', 'electra', 'text_encoder']
        for name in common_names:
            if hasattr(model, name):
                return list(getattr(model, name).named_parameters())

        raise AttributeError(f"找不到模型类型: {model_type}")




def get_other_component_parameters(model):
    """
    获取除了BERT相关的参数外的其他参数
    """
    bert_related_names = ['bert', 'roberta']  # Common names for BERT-related components
    other_parameters = []

    for name, param in model.named_parameters():
        # Exclude parameters belonging to BERT-related components
        if not any(bert_name in name for bert_name in bert_related_names):
            other_parameters.append((name, param))

    return other_parameters



def create_parameter_groups(bert_parameters, other_parameters, no_decay, args):
    """创建优化器参数组"""

    # 过滤掉冻结的参数
    bert_trainable = [(n, p) for n, p in bert_parameters if p.requires_grad]
    other_trainable = [(n, p) for n, p in other_parameters if p.requires_grad]

    optimizer_grouped_parameters = [
        # 其他组件参数 - 需要权重衰减
        {
            "params": [p for n, p in other_trainable if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "param_group_name": "other_with_decay"
        },
        # 其他组件参数 - 不需要权重衰减
        {
            "params": [p for n, p in other_trainable if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr,
            "param_group_name": "other_no_decay"
        },
        # BERT参数 - 需要权重衰减
        {
            "params": [p for n, p in bert_trainable if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr,
            "param_group_name": "bert_with_decay"
        },
        # BERT参数 - 不需要权重衰减
        {
            "params": [p for n, p in bert_trainable if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.bert_lr,
            "param_group_name": "bert_no_decay"
        },
    ]

    # 打印参数组信息
    for i, group in enumerate(optimizer_grouped_parameters):
        param_count = len(group["params"])
        print(f"参数组 {i} ({group['param_group_name']}): {param_count} 个参数, "
              f"学习率: {group['lr']}, 权重衰减: {group['weight_decay']}")

    return optimizer_grouped_parameters


# 可选：更灵活的参数冻结策略
def freeze_params_by_layer(model, freeze_layers=None, freeze_embeddings=True):
    """
    按层冻结参数的更灵活方法

    Args:
        model: 模型
        freeze_layers: 要冻结的层列表，例如 ['layer.0', 'layer.1']
        freeze_embeddings: 是否冻结embedding层
    """
    if freeze_layers is None:
        freeze_layers = []

    frozen_count = 0

    for name, param in model.named_parameters():
        should_freeze = False

        # 冻结指定层
        if any(layer in name for layer in freeze_layers):
            should_freeze = True

        # 冻结embedding层
        if freeze_embeddings and 'embedding' in name.lower():
            should_freeze = True

        # 冻结预训练的视觉编码器
        if 'image_model' in name or 'visual_encoder' in name:
            should_freeze = True

        if should_freeze:
            param.requires_grad = False
            frozen_count += 1

    print(f"按策略冻结了 {frozen_count} 个参数")


# 使用示例
def example_usage():
    """使用示例"""
    # 如果需要更精细的控制，可以这样使用：

    # 1. 冻结BERT的前几层
    # freeze_params_by_layer(model, freeze_layers=['layer.0', 'layer.1', 'layer.2'])

    # 2. 只冻结embedding和图像编码器
    # freeze_params_by_layer(model, freeze_layers=[], freeze_embeddings=True)

    # 3. 渐进式解冻（在训练过程中逐步解冻层）
    # 可以在训练的不同阶段调用不同的冻结策略
    pass


#==============================  =新改进的优化器代码结束==========================================

def prepare_optimizer_scheduler_complete(args, model, training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight","LayerNorm.bias"]

    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()

    other_parameters = list(model.txt2uni.named_parameters()) \
                       + list(model.span_layer.named_parameters()) \
                       + list(model.span_classifier.named_parameters()) \
                       +list(model.txt2img_attention.named_parameters()) \
                       # + list(model.img2txt_attention.named_parameters()) \
                       # + list(model.oov_reg.named_parameters()) \
                       # + list(model.bn_loss.named_parameters())


    #freeze resnet
    for name,param in model.named_parameters():
        if "image_model" in name:
            param.required_grad=False

    #参考：https://blog.csdn.net/winter2121/article/details/116235872
    optimizer_grouped_parameters = [
        # other params
        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr},

        # bert params
        {"params": [p for n, p in bert_parameters if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},

        {"params": [p for n, p in bert_parameters if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )

    return optimizer, scheduler
def prepare_optimizer_scheduler(args, model, training_steps):
    """Unified optimizer and scheduler preparation."""
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # Collect all model parameters
    all_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # Freeze resnet parameters
    # for name, param in model.named_parameters():
    #     if "image_model" in name:
    #         param.requires_grad = False

    # Group parameters for optimizer
    optimizer_grouped_parameters = [
        # Parameters with weight decay
        {"params": [p for n, p in all_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.lr},

        # Parameters without weight decay (e.g., biases and LayerNorm)
        {"params": [p for n, p in all_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr},
    ]

    # Initialize optimizer and scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio*training_steps, num_training_steps=training_steps
    )

    return optimizer, scheduler


def prepare_optimizer_scheduler_tiered(args, model, training_steps):
    """
    PGMNER改进: RoBERTa(名称含 bert) / image_model(clip) / 其余 三组学习率，减轻 large 全任务同 lr 不稳定。
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    tlr = (
        float(args.text_encoder_lr)
        if getattr(args, "text_encoder_lr", None) is not None
        else float(args.bert_lr)
    )
    vlr = (
        float(args.vision_encoder_lr)
        if getattr(args, "vision_encoder_lr", None) is not None
        else 2e-6
    )
    mlr = (
        float(args.main_lr)
        if getattr(args, "main_lr", None) is not None
        else float(args.lr)
    )

    text_params = []
    vision_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "image_model" in n:
            vision_params.append((n, p))
        elif "bert" in n or "roberta" in n.lower():
            text_params.append((n, p))
        else:
            other_params.append((n, p))

    def _make_groups(trainable, lr):
        g = []
        wd_p = [p for n, p in trainable if not any(nd in n for nd in no_decay)]
        nwd_p = [p for n, p in trainable if any(nd in n for nd in no_decay)]
        if wd_p:
            g.append({"params": wd_p, "weight_decay": args.weight_decay, "lr": lr})
        if nwd_p:
            g.append({"params": nwd_p, "weight_decay": 0.0, "lr": lr})
        return g

    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(_make_groups(text_params, tlr))
    optimizer_grouped_parameters.extend(_make_groups(vision_params, vlr))
    optimizer_grouped_parameters.extend(_make_groups(other_params, mlr))

    if not optimizer_grouped_parameters:
        raise ValueError("prepare_optimizer_scheduler_tiered: no trainable parameters")

    optimizer = AdamW(optimizer_grouped_parameters, lr=mlr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * training_steps),
        num_training_steps=training_steps,
    )
    return optimizer, scheduler


def model_save(args, output_dir, model, tokenizer):
    # output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


# save predictions result
def predictions_save(origin_file, predictions, output_file, labels):
    
    pred_label_idx = [x['pred_label_idx'].tolist() for x in predictions]
    all_span_idxs = [x['all_span_idxs'].tolist() for x in predictions]
    span_label_ltoken = [x['span_label_ltoken'].tolist() for x in predictions]
    
    # idx2label = {0:'O', 1:'ORG', 2:'PER', 3:'LOC', 4:'MISC'}
    idx2label = {k: v for k, v in enumerate(labels)}

    with open(output_file, "w") as writer:
        with open(origin_file, "r") as f:
            words = []
            lines = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append(words)
                        words = []
                elif "IMGID" in line:continue
                else:
                    splits = line.split()
                    if len(splits) not in [2, 4]:
                        print('line segs num {}'.format(len(splits)))
                        print(line)
                    words.append(splits[0])
            if words:
                lines.append(words)

            cnt = 0
            for i in range(len(pred_label_idx)):
                for span_idxs, lps, lts in zip(all_span_idxs[i], pred_label_idx[i], span_label_ltoken[i]):
                    word_list = (lines[cnt])[:]

                    for sid, lp,lt in zip(span_idxs, lps, lts):
                        if lp != 0 or lt != 0:
                            plabel = idx2label[int(lp)]
                            tlabel = idx2label[int(lt)]
                            sidx, eidx = sid
                            for k in range(int(sidx), int(eidx) + 1):
                                word_list[k] = word_list[k] + ' ' + tlabel + ' ' + plabel
                    for k in range(len(word_list)):
                        if ' ' not in word_list[k]:
                            word_list[k] = word_list[k] + ' O O'
                                         
                    text = '\n'.join(word_list)
                    writer.write(text + '\n\n')
                    cnt += 1
            
            f.close()
        writer.close()

