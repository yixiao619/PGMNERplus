
from torch import nn
import torch
import copy
import math
import sys
from typing import Tuple
from torch import nn, Tensor, device
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import CLIPConfig,CLIPModel
from .MI_estimators import VIB_M, vCLUB, InfoNCE
from .classifier import MultiNonLinearClassifier
from .model_utils import span_select
from .span_layer import SpanLayer
from loader4image import ViTImageModel
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from models.bottleneck import Bottleneck_Attention
from models.losses.SupConLoss import SupConLoss,SupConLoss_Enhanced,SupConR
from einops import rearrange, repeat
from OT import OptimalTransportFusion
from models.semantic_ot_aligner import SemanticOTAligner
from utils.datasets import get_labels as get_dataset_labels
from MultimodalFusion import MultiModalFusion
from transformers import RobertaTokenizer, RobertaModel
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM



ANNEALING_RATIO = 0.3
SMALL = 1e-08
SAMPLE_SIZE = 5

# span embedding settings
MAX_SPAN_LEN = 6
MORPH_NUM = 5
MAX_SPAN_NUM = 502
TOKEN_LEN_DIM = 50
SPAN_LEN_DIM = 50
SPAN_MORPH_DIM = 100
BOTTLE_NECK_DIM = 100


def ppot_regularizer(ot_plan, labels=None):
    """
    Simplified probability-polarization regularizer on the OT plan γ (static margins, no label-aware intra/inter yet).

    ot_plan: [B, n_text, n_img] OT coupling
    labels: reserved for future intra/inter masking (unused in this toy version)
    """
    del labels  # reserved for label-aware PPOT
    gamma = ot_plan
    with torch.no_grad():
        center = gamma.mean()
        delta_minus = 0.5 * center
        delta_plus = 1.5 * center

    term = (gamma - delta_minus) * (gamma - delta_plus)
    pp_mask = (term < 0).float()
    pp_loss = (pp_mask * term.abs()).mean()
    return pp_loss


def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_attention_map = []
        for layer_module in self.layer:
            s1_hidden_states,attention_map = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
                all_attention_map.append(attention_map)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
            all_attention_map.append(attention_map)
        return all_encoder_layers,all_attention_map

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output,attention_map = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,attention_map

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output,attention_map = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output,attention_map

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        # self.num_attention_heads=6
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer,attention_probs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DistEstimate(nn.Module):
    '''
     refers the implementation of https://github.com/mfederici/Multi-View-Information-Bottleneck/blob/master/utils/modules.py
    '''
    def __init__(self, x_dim, hidden_dim, out_dim):
        super(DistEstimate, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.x_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim*2)
        )

    def forward(self, x):
        params = self.net(x)

        mu, sigma = params[:, :self.out_dim], params[:, self.out_dim: ]
        sigma = F.softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(mu, sigma), 1)

class IBEstimator(nn.Module):
    def __init__(self, size1, size2, hidden_dim):
        super(IBEstimator,self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )

    # Gradient for JSD mutual information estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -F.softplus(-pos).mean() - F.softplus(neg).mean()


class DistEstimate_2(nn.Module):
    """
    分布估计器 - 估计单一模态的特征分布
    通常用于估计 p(z_t) 和 p(z_v)
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)
        return mu, logvar


class DistEstimate_PP_IB(nn.Module):
    """
    概率极化信息瓶颈（PP-IB，PPOT）：VIB 重参数得到 Z 后，用门控 gamma 极化；
    L_polar = sum |min((gamma-0.2)(gamma-0.8), 0)|；Z_final = gamma * Z（逐元素）。
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.polar_gate = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)
        eps = torch.randn_like(logvar)
        z = mu + torch.exp(0.5 * logvar) * eps
        gamma = self.polar_gate(mu)
        z_final = gamma * z
        quad = (gamma - 0.2) * (gamma - 0.8)
        L_polar = torch.sum(
            torch.abs(torch.min(quad, torch.zeros_like(quad)))
        )
        return mu, logvar, z_final, L_polar


class IBEstimator_2(nn.Module):
    """
    信息瓶颈估计器 - 估计两个模态之间的互信息
    实现 I(Z_t; Z_v) 的估计
    """

    def __init__(self, text_dim, visual_dim, hidden_dim):
        super().__init__()
        # 用于估计联合分布和边缘分布
        self.joint_net = nn.Sequential(
            nn.Linear(text_dim + visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # MINE (Mutual Information Neural Estimation) 相关组件
        self.ma_et = None  # 移动平均
        self.ma_rate = 0.01

    def forward(self, z_t, z_v):
        """
        使用MINE方法估计互信息
        """
        # 联合分布：真实的(z_t, z_v)配对
        joint_samples = torch.cat([z_t, z_v], dim=-1)
        joint_scores = self.joint_net(joint_samples)

        # 边缘分布：随机重排的(z_t, z_v)配对
        z_v_shuffled = z_v[torch.randperm(z_v.size(0))]
        marginal_samples = torch.cat([z_t, z_v_shuffled], dim=-1)
        marginal_scores = self.joint_net(marginal_samples)

        # MINE估计
        mi_estimate = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)))

        return mi_estimate, joint_scores, marginal_scores

class BertSpanNerBN(nn.Module):
    def __init__(
        self,
        config,
        args=None,
        num_labels=None
    ):
        # super(BertSpanNerBN, self).__init__(config)
        super().__init__()
        self.args = args
        # ---------------- For Only Test ------------------
        #For saved weights
        self.txt_representations_ori=[]
        self.img_representations_ori=[]
        self.txt_representations_IB = []
        self.img_representations_IB = []
        self.txt_representations_OT = []
        self.img_representations_OT = []
        self.txt_representations_OT_g = []
        self.img_representations_OT_g = []
        self.fusion_text_representations=[]
        self.transport_plans=[]
        # self.objects_representations=[]
        # self.span_representations=[]
        self.span_indexs = []
        self.span_labels=[]
        self.span_mask=[]
        # self.img_span_representations=[]
        # self.txt_span_representations = []


        # ------------------------ Embedding Dims ------------------------------
        # self.txt_dim_global=args.txt_global_dim
        # self.img_dim_global=args.img_global_dim
        # self.txt_dim_local=args.txt_local_dim
        # self.img_dim_local=args.img_local_dim
        # self.hidden_dim=args.hidden_dim

        # ------------------------ Encoder ------------------------------
        ###-----------------For Roberta-----------------
        self.tokenizer=RobertaTokenizer.from_pretrained(args.model_name_or_path)
        self.bert = RobertaModel.from_pretrained(args.model_name_or_path,config=config)
        ###-----------------For  BERT-----------------
        # self.bert = BertModel(config)

        # Save locally
        # self.bert.save_pretrained("../pretrain/roberta-base")
        # self.tokenizer.save_pretrained("../pretrain/roberta-base")
        # self.bert.to(args.device)
        # self.bert.eval()

        self.image_model=ViTImageModel(args)
        # self.img2uni=nn.Linear(args.img_global_dim,args.img_global_dim)
        self.txt2uni=nn.Linear(args.txt_local_dim,args.hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.n_class = 5 if not num_labels else num_labels

        # ---------------- span layer ------------------
        self.span_layer = SpanLayer(config.hidden_size, TOKEN_LEN_DIM, SPAN_LEN_DIM,
                                    SPAN_MORPH_DIM, MAX_SPAN_LEN, MORPH_NUM)

        # SpanLayer 输出维 = EndpointSpanExtractor + spanLen + morph（与 config.hidden_size 有关，不能用固定的 txt_local_dim）
        span_feat_dim = (
            self.span_layer.span_extractor.get_output_dim()
            + self.span_layer.spanLen_embedding.embedding_dim
            + self.span_layer.morph_embedding.embedding_dim
        )
        # CAC：原版 OptimalTransportFusion；Semantic OT；或 GOT（语义 + GW 结构项，互斥）
        self.use_got = getattr(args, "use_got", False)
        self.use_semantic_ot = getattr(args, "use_semantic_ot", False) and not self.use_got
        if self.use_got:
            self.optimal_transport_fusion = None
            self.semantic_ot_aligner = SemanticOTAligner(
                span_feat_dim,
                args.img_local_dim,
                args.hidden_dim,
                eps=getattr(args, "ot_eps", 0.1),
                sinkhorn_iter=getattr(args, "ot_sinkhorn_iter", 10),
                lambda_align=getattr(args, "ot_lambda_align", 1.0),
                use_structure=True,
                lambda_sem=getattr(args, "got_lambda_sem", 0.5),
                struct_mode=getattr(args, "got_struct_mode", "full"),
            )
        elif self.use_semantic_ot:
            self.optimal_transport_fusion = None
            self.semantic_ot_aligner = SemanticOTAligner(
                span_feat_dim,
                args.img_local_dim,
                args.hidden_dim,
                eps=getattr(args, "ot_eps", 0.1),
                sinkhorn_iter=getattr(args, "ot_sinkhorn_iter", 10),
                lambda_align=getattr(args, "ot_lambda_align", 1.0),
                use_structure=False,
            )
        else:
            self.semantic_ot_aligner = None
            # 使用仓库根目录 OT.py（Prompt Bucket + log-Sinkhorn）；仅支持 fusion_method='sinkhorn'
            _use_got_cac = getattr(args, "use_got_cac", False)
            _lg = float(getattr(args, "cac_got_lambda", 0.3)) if _use_got_cac else 0.0
            _go = int(getattr(args, "cac_got_outer_iters", 3)) if _use_got_cac and _lg > 0 else 1
            self.optimal_transport_fusion = OptimalTransportFusion(
                span_feat_dim,
                args.img_local_dim,
                args.hidden_dim,
                fusion_method="sinkhorn",
                temperature=0.1,
                lambda_got=_lg,
                got_outer_iters=_go,
            )



        # 融合层
        self.fusion_text = nn.Linear( args.hidden_dim* 2, args.hidden_dim)
        self.fusion_image = nn.Linear(args.hidden_dim * 2, args.hidden_dim)


        # 层归一化
        self.ln_text = nn.LayerNorm(args.hidden_dim)
        self.ln_image = nn.LayerNorm(args.hidden_dim)


        # 与 pooler / sentence 全局向量维一致：large=1024，base=768
        # baseline：DistEstimate_2；可选 --use_pp_ib 启用 PP-IB
        self.use_pp_ib = bool(getattr(args, "use_pp_ib", False))
        if self.use_pp_ib:
            self.dist_t = DistEstimate_PP_IB(config.hidden_size, 768, 768)
            self.dist_v = DistEstimate_PP_IB(768, 768, 768)
        else:
            self.dist_t = DistEstimate_2(config.hidden_size, 768, 768)
            self.dist_v = DistEstimate_2(768, 768, 768)
        self.tv_ib = IBEstimator(768, 768, 256)


        self.span_classifier = MultiNonLinearClassifier(
           args.hidden_dim, self.n_class, dropout_rate=0.2
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.suploss = SupConR()
        _lab = get_dataset_labels(args.labels)
        _use_mw = getattr(args, "use_misc_weighted_scl", False)
        _mw = getattr(args, "scl_misc_neg_weight", 2.0)
        try:
            _im = _lab.index("MISC")
            _io = _lab.index("ORG")
            _il = _lab.index("LOC")
        except ValueError:
            _use_mw = False
            _im, _io, _il = -1, -1, -1
        self.suploss = SupConLoss(
            use_misc_weighted_scl=_use_mw,
            scl_misc_neg_weight=_mw,
            id_misc=_im,
            id_org=_io,
            id_loc=_il,
        )
        # Span CE：可选轻量 focal（仅 gold-MISC：(1-pt)^gamma * CE；其余标准 CE）
        self.use_misc_focal_loss = bool(
            getattr(args, "use_misc_focal_loss", False) and _im >= 0
        )
        _mfa = float(getattr(args, "misc_focal_alpha", -1.0))
        self.misc_focal_use_alpha = _mfa >= 0.0
        self.misc_focal_alpha = _mfa if self.misc_focal_use_alpha else -1.0
        self.misc_focal_gamma = float(getattr(args, "misc_focal_gamma", 1.0))
        self.misc_focal_ce_scale_other = float(
            getattr(args, "misc_focal_ce_scale_other", 1.0)
        )
        self.id_misc_focal = _im



    def get_mu_logvar_img(self,x):
        mu=self.mu_img(x)
        logvar=self.logvar_img(x)
        logvar=nn.functional.softplus(logvar) + 1e-7

        return mu,logvar

    def reparameterise(self,mu,logvar):
        eps = torch.randn_like(logvar)

        z = mu + torch.exp(0.5 * logvar) * eps

        return z

    def kl_divergence(self, mu, logvar):
        """计算KL散度 KL(q(z|x) || p(z))，其中p(z) = N(0,I)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(self, ori_feas,only_test=False):
        """
        默认有 labels 为 train, 无 labels 为 test
        """


        if only_test:
            self.span_indexs.append(ori_feas['span_token_idxes'].cpu().detach())
            self.span_labels.append(ori_feas['span_labels'].cpu().detach())
            self.span_mask.append(ori_feas['span_masks'].cpu().detach())

        #ori_encoding, cont_encoding = {}, {}
        ori_encoding = {}
        ori_encoding['spans_rep'], ori_encoding['txt_rep_g'], attn_last = self.span_encoding(**ori_feas)
        text_embed_local=ori_encoding['spans_rep']
        if only_test:
            self.txt_representations_ori.append(ori_encoding['spans_rep'].cpu().detach())


        if self.use_pp_ib:
            mu_t_dist, logvar_t_dist, t, L_polar_t = self.dist_t(ori_encoding['txt_rep_g'])
        else:
            mu_t_dist, logvar_t_dist = self.dist_t(ori_encoding['txt_rep_g'])
            t = self.reparameterise(mu_t_dist, logvar_t_dist)
            L_polar_t = None
        if only_test:
            self.txt_representations_IB.append(t.cpu().detach())

        images=ori_feas['image']
        ori_encoding['imag_rep_g'],ori_encoding['imag_rep_l']=self.image_model(images)
        if only_test:
            self.img_representations_ori.append(ori_encoding['imag_rep_l'].cpu().detach())

        if self.use_pp_ib:
            mu_v_dist, logvar_v_dist, v, L_polar_v = self.dist_v(ori_encoding['imag_rep_g'])
        else:
            mu_v_dist, logvar_v_dist = self.dist_v(ori_encoding['imag_rep_g'])
            v = self.reparameterise(mu_v_dist, logvar_v_dist)
            L_polar_v = None
        if only_test:
            self.img_representations_IB.append(v.cpu().detach())
        # v = self.img2uni(v)


        vis_embed_local=ori_encoding['imag_rep_l']

        loss_ot = None
        ot_components = None
        got_loss_cac = None
        if self.use_got or self.use_semantic_ot:
            mask_t = ori_feas['span_masks'].float()
            attn_t = attn_last if self.use_got else None
            (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                transport_plan,
                loss_ot,
                ot_components,
            ) = self.semantic_ot_aligner(
                text_embed_local,
                vis_embed_local,
                mask_t=mask_t,
                mask_v=None,
                attn_t=attn_t,
                span_token_idxes=ori_feas.get("span_token_idxes"),
                patch_pos=None,
            )
        else:
            (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                transport_plan,
                got_loss_cac,
            ) = self.optimal_transport_fusion(
                text_embed_local,
                vis_embed_local,
            )

        # 供可视化 / 调试：最近一次 batch 的传输计划（CAC 与 OT 路径统一）
        self.last_transport_plan = transport_plan.detach().cpu()

        if only_test:
            self.transport_plans.append(transport_plan.cpu().detach())
            text_proj_g=text_proj.mean(dim=1)
            self.txt_representations_OT_g.append(text_proj.cpu().detach())
            self.txt_representations_OT.append(ori_encoding['txt_rep_g'].cpu().detach())
            align_text_g = align_text.mean(dim=1)
            self.img_representations_OT_g.append(visual_proj.cpu().detach())
            self.img_representations_OT.append(ori_encoding['imag_rep_g'].cpu().detach())


        # 多模态融合
        # fused_text, attn_weights = self.multimodal_fusion(text_embed_local, fused_text, fused_visual,)
        # 特征融合
        # 文本特征融合 (原始 + 对齐后的图像特征)
        text_combined = torch.cat([text_proj, align_text], dim=-1)
        fused_text = self.fusion_text(text_combined)
        fused_text = self.ln_text(fused_text + align_text)
        # if only_test:
        #     self.fusion_text_representations.append(fused_text.cpu().detach())
        #     # self.img_representations_OT.append(align_visual.cpu().detach())


        # 图像特征融合 (原始 + 对齐后的文本特征)
        # image_combined = torch.cat([visual_proj, align_visual], dim=-1)
        # fused_image = self.fusion_image(image_combined)
        # fused_image = self.ln_image(fused_image + align_visual)


        # span_fusion_rep=text_sequence_output
        # span_fusion_rep=torch.cat([text_embed_local,text_fused],dim=-1)

        logits = self.span_classifier(fused_text)
        bias = getattr(self, "_misc_logit_bias", None)
        if bias is not None and self.id_misc_focal >= 0:
            logits = logits.clone()
            logits[:, :, self.id_misc_focal] = logits[:, :, self.id_misc_focal] + float(
                bias
            )
        ori_encoding["logits"] = logits
        if only_test:
            self.fusion_text_representations.append(fused_text.cpu().detach())
            # self.img_representations_OT.append(align_visual.cpu().detach())

        # fusion_s = F.normalize(fused_text, p=2, dim=-1)
        fusion_s=fused_text
        if len(fusion_s.size()) < 3:
            fusion_s = repeat(fusion_s, 'b e -> r b e', r=1)
        fusion_s = repeat(rearrange(fusion_s, 'r b e -> (r b) e'), 'b e ->b v e', v=1)

        # ori_encoding['logits'] = self.span_classifier(span_fusion_rep)

        loss_dict = {}
        outputs = [self.softmax(logits)]



        if ori_feas['span_labels'] is not None:
            # TODO, test stop gradient of cont features
            # cont_encoding['spans_rep'].detach()
            loss_dict = self.compute_loss(
                ori_feas,
                ori_encoding,
                t,
                v,
                mu_t_dist,
                logvar_t_dist,
                mu_v_dist,
                logvar_v_dist,
                fusion_s=fusion_s,
                transport_plan=transport_plan,
                loss_ot=loss_ot,
                ot_components=ot_components,
                L_polar_t=L_polar_t,
                L_polar_v=L_polar_v,
                got_loss_cac=got_loss_cac,
            )
            # loss_dict = self.compute_loss(ori_feas, ori_encoding)

        return outputs, loss_dict  # (loss), scores

    def span_encoding(self, input_ids=None, input_mask=None, segment_ids=None,
                      span_token_idxes=None, span_lens=None, morph_idxes=None, **kwargs):
        """
        Encode tokens by Bert, and get span representations.
        """
        # encoder [batch, seq_len, hidden]；GOT 需要最后一层 self-attention 构图
        need_attn = getattr(self, "use_got", False)
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            output_attentions=need_attn,
        )

        sequence_output_global = sequence_output.last_hidden_state[0]  # (seq_len, hidden_size)
        # RoBERTa没有pooler_output，使用<s> token的输出代替
        # pooled_output = output[0]  # 第一个token (<s>) 的表示

        sequence_output_global = sequence_output.pooler_output
        sequence_output_context = sequence_output.last_hidden_state

        # sequence_output_context = sequence_output.last_hidden_state[1:-1]


        span_rep = self.span_layer( sequence_output_context, span_token_idxes.long(), span_lens, morph_idxes)

        attn_last = None
        if need_attn and getattr(sequence_output, "attentions", None) is not None:
            attn_last = sequence_output.attentions[-1]

        return span_rep, sequence_output_global, attn_last

    def compute_loss(self, ori_feas, ori_encoding,t=None,v=None,mu_t_dist=None, logvar_t_dist=None,mu_v_dist=None, logvar_v_dist=None,fusion_s=None, transport_plan=None, loss_ot=None, ot_components=None, L_polar_t=None, L_polar_v=None, got_loss_cac=None):
        """
        计算loss, including: task loss,scl loss,mvib loss
        """
        # ----------compute span classification loss----------
        # loss_dic = {'c': self.compute_clas_loss(ori_feas, ori_encoding),
        #             # 'cc': self.compute_clas_loss(cont_feas, cont_encoding)
        #             }
        _clas = self.compute_clas_loss(ori_feas, ori_encoding, mode="logits")
        if isinstance(_clas, tuple):
            loss_dic = {"c": _clas[0], "c_misc": _clas[1], "misc_span_count": _clas[2]}
        else:
            loss_dic = {"c": _clas}

        # ----------compute multimodal supervised contrastive learning loss----------
        labels = ori_feas['span_labels'].view(-1)
        loss_dic["scl"] = self.suploss(fusion_s, labels)
        if getattr(self.args, "use_misc_weighted_scl", False):
            loss_dic["scl_misc"] = loss_dic["scl"]


        # #****************T和V之间的multi-view 信息瓶颈************
        mi_tv = self.tv_ib(v, t)
        mi_vt=self.tv_ib(t,v)
        mi = (mi_tv + mi_vt) / 2.0
        # # # v_prob = img_dist.log_prob(v).exp()
        # # kl_vt = (img_dist.log_prob(v) - text_dist.log_prob(v))
        # # kl_tv = (text_dist.log_prob(t) - img_dist.log_prob(t))
        # # skl_tv = (kl_tv + kl_vt).mean() / 2.
        # # loss_dic['loss_ib_tv'] = -mi + skl_tv
        #
        # ****************分别计算文本和图像的信息瓶颈************
        # 使用使用DistEstimate_2
        # mi, _, _ = self.tv_ib(t, v)

        # 计算KL散度（用于VAE训练）
        kl_t = self.kl_divergence(mu_t_dist, logvar_t_dist)
        kl_v = self.kl_divergence(mu_v_dist, logvar_v_dist)
        skl_tv = (kl_t + kl_v) / 2.0
        #
        loss_dic['loss_ib_tv'] = 0.001*skl_tv

        # ----------Semantic OT / GOT 显式损失（替代 CAC 时）----------
        # 总 loss 只加 ot_total = ot_alpha * L_ot；L_sem/L_str/L_reg 为分解项（未乘权重）
        if (
            (getattr(self.args, "use_semantic_ot", False) or getattr(self.args, "use_got", False))
            and loss_ot is not None
        ):
            ot_alpha = getattr(self.args, "ot_alpha", 0.05)
            loss_dic["ot_total"] = ot_alpha * loss_ot
            if getattr(self.args, "use_got", False):
                loss_dic["ot_got"] = loss_dic["ot_total"]
            if ot_components is not None:
                loss_dic["ot_sem"] = ot_components["L_sem"]
                loss_dic["ot_reg"] = ot_components["L_reg"]
                if getattr(self.args, "use_got", False):
                    loss_dic["ot_str"] = ot_components["L_str"]

        # ----------PPOT: polarize OT plan (optional)----------
        if getattr(self.args, "use_ppot", False) and transport_plan is not None:
            pp_loss = ppot_regularizer(transport_plan, labels=None)
            loss_dic["ppot"] = self.args.ppot_beta * pp_loss

        # ----------PP-IB：概率极化项 β · L_polar（PPOT）----------
        if L_polar_t is not None and L_polar_v is not None:
            loss_dic["loss_polar"] = self.args.beta * (L_polar_t + L_polar_v)

        # ----------GOT-A（CAC）：需 --use_got_cac；否则与 baseline 一致不加此项----------
        if (
            got_loss_cac is not None
            and getattr(self.args, "use_got_cac", False)
            and not getattr(self.args, "use_semantic_ot", False)
            and not getattr(self.args, "use_got", False)
        ):
            _ga = getattr(self.args, "cac_got_loss_alpha", None)
            _w = _ga if _ga is not None else getattr(self.args, "ot_alpha", 0.05)
            loss_dic["got_cac"] = _w * got_loss_cac

        # ----------compute KL(p(z_o|t), p(z_c|t)) loss----------
        # x_span_cont, y_span_cont = span_select(
        #     ori_encoding['spans_rep'].unsqueeze(1), ori_feas['cont_span_idx'],
        #     cont_encoding['spans_rep'].unsqueeze(1), cont_feas['cont_span_idx']
        # )
        #
        # entity_dist = self.oov_reg.update(x_span_cont, y_span_cont)
        # loss_dic['si'] = self.beta * entity_dist
        #
        # # ----------compute MI(z_o, z_c) loss----------
        # entity_mi = self.gama * self.z_reg(x_span_cont, y_span_cont)
        # loss_dic['gi'] = entity_mi

        # ----------sum loss----------
        # loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])

        # loss_dic['loss']=loss_dic['c']
        # 仅累加主损失项，避免把 ot_sem / ot_reg / ot_str 等分解标量重复计入
        loss_parts = [
            loss_dic["c"],
            loss_dic["scl"],
            loss_dic["loss_ib_tv"],
        ]
        if "ot_total" in loss_dic:
            loss_parts.append(loss_dic["ot_total"])
        if "ppot" in loss_dic:
            loss_parts.append(loss_dic["ppot"])
        if "loss_polar" in loss_dic:
            loss_parts.append(loss_dic["loss_polar"])
        if "got_cac" in loss_dic:
            loss_parts.append(loss_dic["got_cac"])
        loss_dic["loss"] = sum(loss_parts)

        return loss_dic
    def compute_clas_loss(self, features, encoding, mode="logits"):
        """
        可选轻量 focal：仅 gold-MISC 用 (1-pt)^gamma * CE；其余 span 标准 CE。
        misc_focal_alpha>=0 时 MISC 分支再乘 alpha（档位 B）；无 class_weight。
        """
        batch_size, n_span = features["span_labels"].size()
        ori_span_labels = features["span_labels"].view(-1)
        if mode == "logits":
            ori_span_rep = encoding["logits"].view(-1, self.n_class)
        elif mode == "logits_txt":
            ori_span_rep = encoding["logits_txt"].view(-1, self.n_class)
        elif mode == "logits_img":
            ori_span_rep = encoding["logits_img"].view(-1, self.n_class)
        else:
            ori_span_rep = encoding["logits"].view(-1, self.n_class)

        span_weights = features["span_weights"].view(-1)
        span_masks = features["span_masks"].view(-1).bool()

        if not self.use_misc_focal_loss:
            clas_loss = self.cross_entropy(ori_span_rep, ori_span_labels)
            clas_loss = clas_loss.view(batch_size, n_span) * features["span_weights"]
            clas_loss = torch.masked_select(
                clas_loss, features["span_masks"].bool()
            ).mean()
            return clas_loss

        log_probs = F.log_softmax(ori_span_rep, dim=-1)
        log_pt = log_probs.gather(1, ori_span_labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8, max=1.0 - 1e-8)
        ce = -log_pt
        focal_mod = (1.0 - pt).pow(self.misc_focal_gamma)

        is_misc = ori_span_labels == self.id_misc_focal
        misc_term = focal_mod * ce
        if self.misc_focal_use_alpha:
            misc_term = misc_term * float(self.misc_focal_alpha)
        ce_other = ce * float(self.misc_focal_ce_scale_other)
        per = torch.where(is_misc, misc_term, ce_other)
        per = per * span_weights
        clas_loss = torch.masked_select(per, span_masks).mean()

        misc_m = span_masks & is_misc
        n_misc = misc_m.sum()
        if n_misc > 0:
            c_misc = torch.masked_select(per, misc_m).mean().detach()
        else:
            c_misc = per.new_tensor(0.0).detach()
        return clas_loss, c_misc, n_misc.detach()
    def compute_clas_loss_yuan(self, features, encoding):
        """
        计算分类loss.
        """
        batch_size, n_span = features['span_labels'].size()
        ori_span_rep = encoding['logits'].view(-1, self.n_class)
        ori_span_labels = features['span_labels'].view(-1)
        # print(features['span_labels'][0])
        # print(features['span_labels'].shape)
        # print(ori_span_labels.shape)
        # print(encoding['logits'].shape)
        # print(ori_span_rep.shape)
        # #exit()

        clas_loss = self.cross_entropy(ori_span_rep, ori_span_labels)
        clas_loss = clas_loss.view(batch_size, n_span) * features['span_weights']
        clas_loss = torch.masked_select(clas_loss, features['span_masks'].bool()).mean()

        return clas_loss
