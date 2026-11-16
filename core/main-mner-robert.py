# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition
on Twitter2015 and Twitter 2017. """
import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch

sys.path.append('../')

#import fitlogc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
from eval_metric import span_f1_prune
from transformers import RobertaTokenizer, RobertaModel
from engine_utils import *
from utils.datasets import (
    collate_fn,
    get_labels,
    load_aux_knowledge_jsonl,
    load_examples,
    load_vllm_desc_jsonl,
    SpanNerDataset,
)
# from models.bn_bert_ner import BertSpanNerBN #融合了文本和(T，V,V)信息，添加了信息瓶颈与图像融合
# from models.bn_bert_ner_complete import BertSpanNerBN #融合了文本和(T，V,V)信息，但是没有使用信息瓶颈
# from models.bn_bert_ner_token import BertSpanNerBN
# from models.bn_bert_ner_fusion import BertSpanNerBN# #融合了文本和(T，V,V)信息，在图像处添加了信息瓶颈
# from models.bn_bert_ner_txt import BertSpanNerBN
# from models.bn_bert_ner_MVIB import BertSpanNerBN#融合了文本和（T,V,V）信息，并计算了文本与图像之间的信息瓶颈

from models.bn_bert_ner_OT_robert import BertSpanNerBN

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean, got {!r}".format(v))


def get_args():
    parser = arg_parse()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        # default="/root/MINER2/MINER/data/WNUT2017/",
        default="data/twitter2017",
        type=str,
        help="The input data dir. Should contain the training files for the "
             "Multimodal NER task.",
    )

    parser.add_argument(
        "--output_dir",
        default="out/twitter2017/roberta_large",
        type=str,
        help="The output directory where the model predictions and "
             "checkpoints will be written.",
    )

    parser.add_argument(
        '--vit_name',
        default='pretrain/clip-vit-base-patch16',
        type=str,
        help="CLIP checkpoint dir or Hugging Face id (e.g. openai/clip-vit-base-patch16)",
    )

    # fitlog debug settings, --debug for True, otherwise for False
    # parser.add_argument("--debug", action="store_true",
    #                     help="Whether record results and params.")

    # Other parameters
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="GPU number id")

    parser.add_argument(
        "--epoch", default=30, type=float,
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--beta", default=0.001, type=float,
        help="weights of oov regular"
    )

    parser.add_argument(
        "--gama", default=0.01, type=float,
        help="weights of oov regular"
    )

    parser.add_argument(
        "--beta_txt", default=0.001, type=float,
        help="weights of InfoNCE"
    )

    # 0 means typos， 1 means switch
    # parser.add_argument(
    #     "--switch_ratio", default=0.5, type=float,
    #     help="Entity switch ratio."
    # )

    # training parameters
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--freeze_pretrained", default=False, type=_str2bool,
                        help="Whether to freeze the parameters of pre-trained language model.")

    parser.add_argument("--patience", default=6, type=int)
    parser.add_argument("--do_train", default=True, type=_str2bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=_str2bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=True, type=_str2bool,
                        help="Whether to run predictions on the test set.")
    # TODO, try different seed
    parser.add_argument("--seed", type=int, default=666,
                        help="random seed for initialization")

    parser.add_argument(
        "--use_ppot",
        action="store_true",
        help="Enable probability-polarized OT regularizer on the OT plan.",
    )
    parser.add_argument(
        "--ppot_beta",
        type=float,
        default=0.1,
        help="Weight of PPOT regularizer term.",
    )

    # Semantic OT（显式 ⟨γ,C⟩ + 熵正则 L_ot）替代原 CAC / Sinkhorn 融合路径
    parser.add_argument(
        "--use_semantic_ot",
        action="store_true",
        help="Use SemanticOTAligner (semantic cost + entropy + Sinkhorn) instead of OptimalTransportFusion.",
    )
    parser.add_argument(
        "--use_got",
        action="store_true",
        help="Graph OT: semantic Wasserstein term + GW structure term (exclusive with --use_semantic_ot).",
    )
    parser.add_argument(
        "--got_lambda_sem",
        type=float,
        default=0.5,
        help="GOT: weight λ on L_sem; structure term uses (1-λ).",
    )
    parser.add_argument(
        "--got_struct_mode",
        type=str,
        default="full",
        choices=("full", "identity", "ones"),
        help="GOT only: full=attn+grid graphs; identity=I for A_t/A_v; ones=all-ones ablation.",
    )
    parser.add_argument(
        "--got_ot_backend",
        type=str,
        default="gw",
        help='With --use_got: "gw" = semantic + GW; else partial_sinkhorn | node_sinkhorn '
        "| partial_only | entropy_only (dust-bin partial OT variants, no GW).",
    )
    parser.add_argument(
        "--partial_mass_init",
        type=float,
        default=0.6,
        help="Initial logit for learnable partial mass s (PartialSinkhorn / partial_only).",
    )
    parser.add_argument(
        "--dust_cost",
        type=float,
        default=0.5,
        help="Dust-bin penalty tau in PartialSinkhorn.",
    )
    parser.add_argument(
        "--ot_alpha",
        type=float,
        default=0.05,
        help="Weight for semantic OT loss L_ot in total loss.",
    )
    parser.add_argument(
        "--use_pp_ib",
        action="store_true",
        help="Enable PP-IB (polarization gate + L_polar). Default off preserves original VIB (DistEstimate_2).",
    )
    parser.add_argument(
        "--use_got_cac",
        action="store_true",
        help="Enable GOT-A on CAC (OT.py): GW fused cost + got_cac loss. Default off preserves original CAC.",
    )
    parser.add_argument(
        "--cac_got_lambda",
        type=float,
        default=0.3,
        help="GOT-A (with --use_got_cac): GW weight λ in C_work = C_node + λ·LC.",
    )
    parser.add_argument(
        "--cac_got_outer_iters",
        type=int,
        default=3,
        help="GOT-A: outer Sinkhorn iterations (only if --use_got_cac and λ>0).",
    )
    parser.add_argument(
        "--cac_got_loss_alpha",
        type=float,
        default=None,
        help="Weight for got_cac; default None uses --ot_alpha.",
    )
    parser.add_argument(
        "--ot_eps",
        type=float,
        default=0.1,
        help="Entropy temperature ε in Sinkhorn / L_ot.",
    )
    parser.add_argument(
        "--ot_sinkhorn_iter",
        type=int,
        default=10,
        help="Sinkhorn iterations for SemanticOTAligner.",
    )
    parser.add_argument(
        "--ot_lambda_align",
        type=float,
        default=1.0,
        help="Scale for gamma @ V when forming align_text.",
    )
    parser.add_argument(
        "--use_misc_weighted_scl",
        action="store_true",
        help="Upweight SCL denominator for MISC<->ORG/LOC negative pairs (MTC boundary).",
    )
    parser.add_argument(
        "--scl_misc_neg_weight",
        type=float,
        default=2.0,
        help="Weight for MISC vs ORG/LOC pairs in weighted SCL denominator.",
    )
    parser.add_argument(
        "--use_misc_focal_loss",
        action="store_true",
        help="Gold-MISC: (1-pt)^gamma*CE (+ optional alpha); other spans: plain CE.",
    )
    parser.add_argument(
        "--misc_focal_gamma",
        type=float,
        default=1.0,
        help="Focal (1-pt)^gamma on gold-MISC spans only.",
    )
    parser.add_argument(
        "--misc_focal_alpha",
        type=float,
        default=-1.0,
        help="If >=0, multiply MISC focal term by this scalar; if <0, no alpha (tier A).",
    )
    parser.add_argument(
        "--misc_focal_ce_scale_other",
        type=float,
        default=1.0,
        help="Multiply standard CE on non-MISC gold spans (tier B e.g. 0.4; default 1=no scaling).",
    )
    parser.add_argument(
        "--beta_polar",
        type=float,
        default=0.01,
        help="PP-IB: weight on L_polar (replaces legacy --beta for polarization term).",
    )
    parser.add_argument(
        "--delta_minus",
        type=float,
        default=0.2,
        help="PP-IB: lower margin on polar gate gamma in L_polar (was fixed 0.2).",
    )
    parser.add_argument(
        "--delta_plus",
        type=float,
        default=0.8,
        help="PP-IB: upper margin on polar gate gamma in L_polar (was fixed 0.8).",
    )
    parser.add_argument(
        "--pp_ib_lambda_balance",
        type=float,
        default=1.0,
        help="PP-IB: weight on batch-mean balance hinge in L_polar (0 disables balance term).",
    )
    parser.add_argument(
        "--beta_polar_warmup_epochs",
        type=int,
        default=3,
        help="PP-IB: linear warmup epochs for beta_polar * L_polar (0 = no warmup).",
    )
    parser.add_argument(
        "--pp_ib_gate_init_std",
        type=float,
        default=0.01,
        help="PP-IB: polar_gate Linear weight init std (0 = PyTorch default init).",
    )
    parser.add_argument(
        "--pp_ib_gate_init_bias",
        type=float,
        default=0.0,
        help="PP-IB: polar_gate Linear bias init (sigmoid(1)≈0.73 for retention start).",
    )
    parser.add_argument(
        "--pp_ib_gate_on_hidden",
        action="store_true",
        help="PP-IB: polar gate reads pre-VIB hidden h instead of mu.",
    )
    parser.add_argument(
        "--pp_ib_probe",
        action="store_true",
        help="Enable PPIBProbe logging for Step-A gate/mu diagnostics.",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=None,
        help="Alias for --ib_kl_weight (VIB KL coefficient).",
    )
    parser.add_argument(
        "--pp_ib_logvar_clamp",
        type=float,
        default=0.0,
        help="PP-IB: clamp logvar to [-v, v] when v>0; 0 disables clamp.",
    )
    # ----- Round 2 (span MNER; no CRF — biases apply to span_classifier output) -----
    parser.add_argument(
        "--span_detection_weight",
        type=float,
        default=0.0,
        help="Aux span entity-vs-O head on fused span reps; 0 disables.",
    )
    parser.add_argument(
        "--span_head_dropout",
        type=float,
        default=0.1,
        help="Dropout inside span entity detection head.",
    )
    parser.add_argument(
        "--crf_misc_bias",
        type=float,
        default=0.0,
        help="Init bias on MISC logit (CRF analogue; 0=no change).",
    )
    parser.add_argument(
        "--crf_entity_bias",
        type=float,
        default=0.0,
        help="Init bias on non-O entity logits except MISC uses crf_misc_bias.",
    )
    parser.add_argument(
        "--crf_o_penalty",
        type=float,
        default=0.0,
        help="Init bias on O logit (typically negative).",
    )
    parser.add_argument(
        "--ib_consistency_weight",
        type=float,
        default=0.0,
        help="Global IB latent paired contrastive loss; 0 disables.",
    )
    parser.add_argument(
        "--ib_cons_temperature",
        type=float,
        default=0.5,
        help="Temperature for IB consistency contrastive.",
    )
    parser.add_argument(
        "--use_hf_clip_vision",
        type=_str2bool,
        default=True,
        help="Load CLIP vision with pretrained weights via CLIPVisionModel. "
        "Default True (the legacy non-HF path is broken and ignored).",
    )
    parser.add_argument(
        "--auto_img_dims",
        type=_str2bool,
        default=True,
        help="Set img_global_dim/img_local_dim from CLIPVisionConfig.from_pretrained(vit_name).",
    )
    parser.add_argument(
        "--ot_warmup_epochs",
        type=int,
        default=0,
        help="Partial OT: force full mass s=1 for this many epochs (relative to ot_start_epoch).",
    )
    parser.add_argument(
        "--ot_start_epoch",
        type=int,
        default=0,
        help="0-based epoch offset before OT warmup counter starts (stage-2 training).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Optional path to .pth state_dict to load before training (strict=False).",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Exponential moving average of weights (needs torch-ema; eval uses EMA params).",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay when --use_ema.",
    )
    parser.add_argument(
        "--got_mode",
        type=str,
        default=None,
        help="Alias for --got_ot_backend when set.",
    )
    parser.add_argument(
        "--entropy_eps",
        type=float,
        default=None,
        help="Alias for --ot_eps (Sinkhorn temperature) when set.",
    )

    parser.add_argument(
        "--visual_suffix_jsonl",
        type=str,
        default="",
        help="JSONL lines: {\"img_id\":\"...\", \"suffix\":\" token1 token2 ...\"} appended as O-label words after each sentence.",
    )
    parser.add_argument(
        "--use_aux_knowledge",
        action="store_true",
        help="Append cached VLLM aux knowledge after tweet (default OFF for exact repro).",
    )
    parser.add_argument(
        "--aux_knowledge_path",
        type=str,
        default="",
        help="JSONL from aux_knowledge/generate_aux.py (keyed by imgid/sample_id).",
    )
    parser.add_argument(
        "--aux_train_only",
        action="store_true",
        help="Append aux knowledge only on train split (dev/test eval without aux).",
    )
    parser.add_argument(
        "--eval_split_for_best",
        type=str,
        default="dev",
        choices=("dev", "test"),
        help="Split used for in-training model selection (default dev).",
    )
    parser.add_argument(
        "--max_aux_words",
        type=int,
        default=12,
        help="Max words appended from compacted aux knowledge.",
    )
    parser.add_argument(
        "--aux_dropout",
        type=float,
        default=0.15,
        help="Train-only: prob to skip aux append (regularization).",
    )
    parser.add_argument(
        "--clas_loss_weight",
        type=float,
        default=1.0,
        help="Span CE weight (default 2.0 when --use_aux_knowledge in model).",
    )
    parser.add_argument(
        "--scl_loss_weight",
        type=float,
        default=1.0,
        help="SCL weight (default 0.15 when --use_aux_knowledge in model).",
    )
    parser.add_argument(
        "--use_vllm_desc",
        action="store_true",
        help="PGMNER+VLLM: append cached image description after tweet ([SEP] style).",
    )
    parser.add_argument(
        "--vllm_desc_path",
        type=str,
        default="",
        help="JSONL cache: data/<dataset>/vllm_desc/{split}.jsonl (supports {split},{dataset}).",
    )
    parser.add_argument(
        "--vllm_desc_train_only",
        action="store_true",
        help="Append descriptions only on train; dev/test eval without desc.",
    )
    parser.add_argument(
        "--desc_max_words",
        type=int,
        default=60,
        help="Max words from VLLM description tail (~30-60 tokens).",
    )
    parser.add_argument(
        "--desc_sep_token",
        type=str,
        default="</s>",
        help="Separator pseudo-word between tweet and description.",
    )
    parser.add_argument(
        "--tweet_max_seq_len",
        type=int,
        default=128,
        help="Token budget for tweet+spans; total --max_seq_len may be larger with desc.",
    )
    parser.add_argument(
        "--desc_fusion",
        type=str,
        default="concat",
        choices=("concat", "separate"),
        help="concat=PGIM text append (default); separate=encode desc alone (optional).",
    )
    parser.add_argument(
        "--vllm_desc_dropout",
        type=float,
        default=0.0,
        help="Train-only prob to skip description append.",
    )
    args = parser.parse_args()
    if args.entropy_eps is not None:
        args.ot_eps = float(args.entropy_eps)
    if args.got_mode:
        args.got_ot_backend = args.got_mode
    if getattr(args, "auto_img_dims", True):
        try:
            from transformers import CLIPVisionConfig

            vc = CLIPVisionConfig.from_pretrained(args.vit_name)
            args.img_global_dim = vc.hidden_size
            args.img_local_dim = vc.hidden_size
            logger.info(
                "auto_img_dims from HF CLIP vision %s: hidden=%s",
                args.vit_name,
                vc.hidden_size,
            )
        except Exception as ex:
            logger.warning("auto_img_dims skipped: %s", ex)
    # Path to a file containing all labels.
    args.labels = os.path.join(args.data_dir, './labels.txt')
    if getattr(args, "kl_weight", None) is not None:
        args.ib_kl_weight = float(args.kl_weight)
    # print(args.labels)
    # Path to a file containing import substring of each category
    # args.pmi_json = os.path.join(args.data_dir, './pmi.json')
    # Path to a file containing entities of each category
    # args.entity_json = os.path.join(args.data_dir, './entity.json')
    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_id)
                device = torch.device("cuda", args.gpu_id)
            else:
                logger.warning("CUDA 不可用，已改用 CPU")
        except RuntimeError as e:
            device = torch.device("cpu")
            logger.warning(
                "无法使用 GPU（驱动过旧、未安装 CUDA 或与 PyTorch 不匹配等）：%s；已改用 CPU。"
                " 需要 GPU 时请升级驱动或安装对应 CUDA 的 PyTorch，或显式使用 --gpu_id -1。",
                e,
            )
    args.device = device

    args.visual_suffix_map = None
    vj = (getattr(args, "visual_suffix_jsonl", None) or "").strip()
    if vj:
        if os.path.isfile(vj):
            m = {}
            with open(vj, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    kid = o.get("img_id", o.get("imgid", ""))
                    if kid is None:
                        continue
                    k = str(kid).strip()
                    suf = o.get("suffix", o.get("visual_context", ""))
                    if k and suf:
                        m[k] = str(suf)
            args.visual_suffix_map = m
            logger.info("visual_suffix_jsonl: loaded %d entries from %s", len(m), vj)
        else:
            logger.warning("visual_suffix_jsonl not found: %s", vj)

    args.aux_knowledge_path = (getattr(args, "aux_knowledge_path", None) or "").strip()
    if getattr(args, "use_aux_knowledge", False) and not args.aux_knowledge_path:
        ds = os.path.basename(args.data_dir.rstrip("/"))
        args.aux_knowledge_path = f"aux_knowledge/{{dataset}}_{{split}}_aux.jsonl"
    args.aux_knowledge_map = None

    return args


def resolve_vllm_desc_map(args, split: str):
    if not getattr(args, "use_vllm_desc", False):
        return None
    if getattr(args, "vllm_desc_train_only", False) and split != "train":
        logger.info("vllm_desc[%s]: skipped (vllm_desc_train_only)", split)
        return None
    vp = (getattr(args, "vllm_desc_path", None) or "").strip()
    ds = os.path.basename(args.data_dir.rstrip("/"))
    if not vp:
        vp = os.path.join(args.data_dir, "vllm_desc", "{split}.jsonl")
    path = vp.format(split=split, dataset=ds) if ("{split}" in vp or "{dataset}" in vp) else vp
    m = load_vllm_desc_jsonl(path)
    logger.info("vllm_desc[%s]: %d entries from %s", split, len(m), path)
    return m


def resolve_aux_knowledge_map(args, split: str):
    if not getattr(args, "use_aux_knowledge", False):
        return None
    if getattr(args, "aux_train_only", False) and split != "train":
        logger.info("aux_knowledge[%s]: skipped (aux_train_only)", split)
        return None
    ak = (getattr(args, "aux_knowledge_path", None) or "").strip()
    if not ak:
        raise ValueError("--use_aux_knowledge requires --aux_knowledge_path")
    ds = os.path.basename(args.data_dir.rstrip("/"))
    path = ak.format(split=split, dataset=ds) if ("{split}" in ak or "{dataset}" in ak) else ak
    m = load_aux_knowledge_jsonl(path)
    logger.info("aux_knowledge[%s]: %d entries from %s", split, len(m), path)
    return m


args = get_args()
if getattr(args, "use_vllm_desc", False) and getattr(args, "use_aux_knowledge", False):
    raise ValueError("Use either --use_vllm_desc or --use_aux_knowledge, not both.")
if getattr(args, "use_vllm_desc", False) and args.max_seq_len <= args.tweet_max_seq_len:
    logger.info(
        "use_vllm_desc: bump max_seq_len %d -> %d for description tail",
        args.max_seq_len,
        args.tweet_max_seq_len + 68,
    )
    args.max_seq_len = args.tweet_max_seq_len + 68
set_seed(args)


# setup fitlog, debug mode, skip recording
# if args.debug is True:
#     fitlog.debug()
#
# fitlog.commit(__file__)  # auto commit your codes
# fitlog.set_log_dir('logs/')  # set the logging directory
# fitlog.add_hyper(args)  # 通过这种方式记录Argu
# mentParser的参数
# fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.warning("Process device: %s", args.device)


def _log_ppib_probe(probe, model) -> None:
    pt = getattr(model.dist_t, "last_probe", None)
    pv = getattr(model.dist_v, "last_probe", None)
    if pt is None or pv is None:
        return
    mu = torch.stack([pt["mu"], pv["mu"]], dim=1)
    gate = torch.cat([pt["gate"], pv["gate"]], dim=-1)
    gate_logit = torch.cat([pt["gate_logit"], pv["gate_logit"]], dim=-1)
    probe.log_step(mu=mu, gate=gate, gate_logit=gate_logit, mask=None)


def train(args, model, tokenizer, labels):
    """ Train the model """
    train_examples = load_examples(
        args.data_dir,
        mode="train",
        tokenizer=tokenizer,
        visual_suffix_map=args.visual_suffix_map,
        aux_knowledge_map=resolve_aux_knowledge_map(args, "train"),
        vllm_desc_map=resolve_vllm_desc_map(args, "train"),
        max_seq_len=args.max_seq_len,
        max_aux_words=int(getattr(args, "max_aux_words", 12)),
        aux_dropout=float(getattr(args, "aux_dropout", 0.0)),
        desc_max_words=int(getattr(args, "desc_max_words", 60)),
        vllm_desc_dropout=float(getattr(args, "vllm_desc_dropout", 0.0)),
        desc_sep_token=str(getattr(args, "desc_sep_token", "</s>")),
    )
    #training_steps = (len(train_examples) - 1 / args.batch_size + 1) * args.epoch
    training_steps=math.ceil(len(train_examples)/args.batch_size)*args.epoch
    # Prepare optimizer and schedule (linear warmup and decay)
    # optimizer, scheduler = prepare_optimizer_scheduler_fusion(args, model, training_steps)
    # optimizer, scheduler = prepare_optimizer_scheduler_complete(args, model, training_steps)
    optimizer, scheduler = prepare_optimizer_scheduler(args, model, training_steps)
    # optimizer, scheduler = prepare_optimizer_scheduler_OT(args, model, training_steps)

    ema = None
    if getattr(args, "use_ema", False):
        try:
            from torch_ema import ExponentialMovingAverage

            ema = ExponentialMovingAverage(
                model.parameters(), decay=float(getattr(args, "ema_decay", 0.999))
            )
            logger.info("EMA enabled decay=%s", getattr(args, "ema_decay", 0.999))
        except ImportError:
            logger.warning(
                "torch-ema not installed (pip install torch-ema); EMA disabled."
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", training_steps)

    probe = None
    if getattr(args, "pp_ib_probe", False) and getattr(model, "use_pp_ib", False):
        from pp_ib_probe import PPIBProbe

        probe = PPIBProbe(
            out_dir=os.path.join(args.output_dir, "probe"),
            beta_polar=float(getattr(args, "beta_polar", 0.01)),
            lambda_balance=float(getattr(args, "pp_ib_lambda_balance", 1.0)),
        )
        logger.info("PPIBProbe enabled -> %s", probe.out_dir)

    global_step = 0
    best_score = 0.0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    logger.info(
        "Building train SpanNerDataset once (%d examples; image I/O + tokenization) — please wait...",
        len(train_examples),
    )
    train_dataset = SpanNerDataset(
        train_examples, args=args, tokenizer=tokenizer, labels=labels
    )
    train_iterator = trange(int(args.epoch), desc="Epoch")
    epoch_num = 0
    n_patience=0

    total_training_time=0.0
    for _ in train_iterator:
        epoch_num += 1
        if hasattr(model, "set_polar_train_epoch"):
            model.set_polar_train_epoch(epoch_num)
        ot_we = int(getattr(args, "ot_warmup_epochs", 0) or 0)
        ot_se = int(getattr(args, "ot_start_epoch", 0) or 0)
        rel_ep = epoch_num - 1 - ot_se
        in_ot_wu = ot_we > 0 and rel_ep >= 0 and rel_ep < ot_we
        if hasattr(model, "set_partial_ot_warmup"):
            model.set_partial_ot_warmup(in_ot_wu)
            _al = getattr(model, "semantic_ot_aligner", None)
            _po = getattr(_al, "_partial_ot", None) if _al is not None else None
            if _po is not None and hasattr(_po, "get_s"):
                try:
                    logger.info(
                        "[OT] epoch=%d warmup=%s partial_s=%.4f",
                        epoch_num,
                        in_ot_wu,
                        float(_po.get_s().item()),
                    )
                except Exception:
                    logger.info(
                        "[OT] epoch=%d warmup=%s",
                        epoch_num,
                        in_ot_wu,
                    )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)

        logger.info("Training epoch num {0}".format(epoch_num))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        misc_focal_num, misc_focal_den = 0.0, 0.0

        for step, batch in enumerate(epoch_iterator):
            start_time=time.time()
            save_flag = False
            model.train()
            ori_fea_tensors = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(ori_fea_tensors)
            if probe is not None:
                _log_ppib_probe(probe, model)

            loss_dic = outputs[1]
            loss = loss_dic['loss']
            loss.backward()


            #fitlog.add_loss(loss.tolist(), name="Loss", step=global_step)
            tr_loss += loss.item()
            if getattr(args, "use_misc_focal_loss", False) and "c_misc" in loss_dic:
                mc = loss_dic.get("misc_span_count")
                if mc is not None and mc.item() > 0:
                    misc_focal_num += loss_dic["c_misc"].item() * mc.item()
                    misc_focal_den += mc.item()
            _skip_bar = frozenset({"loss", "misc_span_count"})
            description = "".join(
                [
                    "{0}:{1}, ".format(k, round(v.item(), 5))
                    for k, v in loss_dic.items()
                    if k not in _skip_bar
                ]
            ).strip(", ")
            epoch_iterator.set_description(description)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            end_time = time.time()
            train_one_bath_time = end_time - start_time
            total_training_time += train_one_bath_time
            global_step += 1

            if global_step % len(train_dataloader) == 0:
                # default evaluate during training
                ev_ctx = (
                    ema.average_parameters()
                    if ema is not None
                    else contextlib.nullcontext()
                )
                eval_split = getattr(args, "eval_split_for_best", "dev") or "dev"
                with ev_ctx:
                    results, _ = evaluate(
                        args,
                        model,
                        tokenizer,
                        labels,
                        mode=eval_split,
                        prefix="{}".format(global_step),
                        epoch_num=epoch_num,
                        split_tag=eval_split,
                    )

                # add fitlog
                #fitlog.add_metric({"dev" + '-' + "f1": results['span_f1']}, step=global_step)

                if best_score <= results['span_f1']:
                    save_flag=True
                    n_patience=0
                    best_score = results['span_f1']
                    # output_dir = os.path.join(args.output_dir, "best_checkpoint")
                    print("best score f1:",best_score)
                    #fitlog.add_best_metric({"dev-f1": results['span_f1']})
                else:
                    n_patience+=1
                    print("n_patience:",n_patience)
                    # print(global_step)
                    # print(len(train_dataloader))
                    #output_dir = args.output_dir
                if save_flag :

                    if ema is not None and hasattr(ema, "copy_to"):
                        ema.copy_to(model.parameters())
                    try:
                        torch.save(model.state_dict(),args.output_dir + "/best_model.pth")
                    finally:
                        if ema is not None and hasattr(ema, "restore"):
                            ema.restore(model.parameters())
                    # model_save(args, args.output_dir, model, tokenizer)
                    logger.info("Save best model at {}".format(args.output_dir))
        if probe is not None:
            probe.epoch_end(epoch_num)
        if getattr(args, "use_misc_focal_loss", False):
            if misc_focal_den > 0:
                logger.info(
                    "Epoch %d: mean focal loss on gold-MISC spans (weighted) = %.6f",
                    epoch_num,
                    misc_focal_num / misc_focal_den,
                )
            else:
                logger.info(
                    "Epoch %d: use_misc_focal_loss on but no gold-MISC spans counted in train",
                    epoch_num,
                )
        if args.patience>0 and n_patience==args.patience:
            break
        # if save_flag:
        #     model_save(args, output_dir, model, tokenizer)
    print("best_score span f1:",best_score)
    print("epoch_num:",epoch_num)
    args._final_epoch = epoch_num
    avrage_train_time=total_training_time/epoch_num
    print("训练one epoch的平均时间为: ",avrage_train_time)

    if probe is not None:
        probe.finalize()

    return global_step, tr_loss / global_step


def _compute_per_class_span_metrics(dev_outputs, labels):
    """Span-level greedy-match per-entity P/R/F1 (+ support from gold)."""
    from collections import defaultdict

    idx2label = {i: lab for i, lab in enumerate(labels)}
    entity_labels = [lab for lab in labels if lab != "O"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    for out in dev_outputs:
        idxs = out["all_span_idxs"].cpu()
        pred = out["pred_label_idx"].cpu()
        gold = out["span_label_ltoken"].cpu()
        mask = out.get("span_masks")
        if mask is not None:
            mask = mask.cpu()
        bsz = pred.shape[0]
        for b in range(bsz):
            ps, gs = set(), set()
            for i in range(pred.shape[1]):
                if mask is not None and mask[b, i].item() <= 0:
                    continue
                ws, we = int(idxs[b, i, 0]), int(idxs[b, i, 1])
                pl, gl = int(pred[b, i]), int(gold[b, i])
                if pl != 0:
                    ps.add((ws, we, idx2label[pl]))
                if gl != 0:
                    gs.add((ws, we, idx2label[gl]))
            for k in gs:
                support[k[2]] += 1
            for k in ps:
                if k in gs:
                    tp[k[2]] += 1
                else:
                    fp[k[2]] += 1
            for k in gs:
                if k not in ps:
                    fn[k[2]] += 1

    def _prf(lab):
        p = tp[lab] / (tp[lab] + fp[lab] + 1e-10)
        r = tp[lab] / (tp[lab] + fn[lab] + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        return round(float(p), 5), round(float(r), 5), round(float(f1), 5), int(support[lab])

    per_class = {}
    for lab in entity_labels:
        p, r, f1, sup = _prf(lab)
        per_class[lab] = {"p": p, "r": r, "f1": f1, "support": sup}
    return per_class


def _append_per_class_history(args, split, epoch_num, prefix, overall_res, per_class):
    out_dir = getattr(args, "output_dir", "") or "."
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "per_class_history.jsonl")
    row = {
        "split": split,
        "epoch": epoch_num,
        "prefix": str(prefix),
        "overall": {
            "p": overall_res.get("span_precision"),
            "r": overall_res.get("span_recall"),
            "f1": overall_res.get("span_f1"),
        },
        "per_class": per_class,
    }
    with open(hist_path, "a", encoding="utf-8") as hf:
        hf.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[per-class] split=%s epoch=%s overall_f1=%s", split, epoch_num, row["overall"]["f1"])
    for lab in sorted(per_class.keys()):
        m = per_class[lab]
        logger.info(
            "[per-class] %s P=%.5f R=%.5f F1=%.5f support=%d",
            lab,
            m["p"],
            m["r"],
            m["f1"],
            m["support"],
        )


def evaluate(
    args,
    model,
    tokenizer,
    labels,
    mode="test",
    prefix="",
    examples=None,
    only_test=False,
    epoch_num=None,
    split_tag=None,
):

    eval_examples = (
        examples
        if examples
        else load_examples(
            args.data_dir,
            mode=mode,
            tokenizer=tokenizer,
            visual_suffix_map=args.visual_suffix_map,
            aux_knowledge_map=resolve_aux_knowledge_map(args, mode),
            vllm_desc_map=resolve_vllm_desc_map(args, mode),
            max_seq_len=args.max_seq_len,
            max_aux_words=int(getattr(args, "max_aux_words", 12)),
            aux_dropout=0.0,
            desc_max_words=int(getattr(args, "desc_max_words", 60)),
            vllm_desc_dropout=0.0,
            desc_sep_token=str(getattr(args, "desc_sep_token", "</s>")),
        )
    )
    eval_dataset = SpanNerDataset(eval_examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)

    # accelerate evaluation speed
    args.eval_batch_size = 8
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    logger.info("***** Running evaluation {0} {1} *****".format(mode, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    dev_outputs = []
    model.eval()    # close drop out, batch normalization
    total_test_time=0.0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ori_fea_tensors = {k: v.to(args.device) for k, v in batch.items()}
        #cont_fea_tensors = {k: v.to(args.device) for k, v in batch[1].items()}
        with torch.no_grad():
            # without labels, direct out tags
            start_time=time.time()
            predicts, _ = model(ori_fea_tensors,only_test)
            end_time=time.time()
            test_time=end_time-start_time
            total_test_time+=test_time
            # span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
            span_f1s, pred_label_idx = span_f1_prune(
                ori_fea_tensors['span_word_idxes'],
                predicts[0],
                ori_fea_tensors['span_labels'],
                ori_fea_tensors['span_masks']
            )
            outputs = {
                'span_f1s': span_f1s,
                'pred_label_idx': pred_label_idx,
                'all_span_idxs': ori_fea_tensors['span_word_idxes'],
                'span_label_ltoken': ori_fea_tensors['span_labels'],
                'span_masks': ori_fea_tensors['span_masks'],
            }
            dev_outputs.append(outputs)

        nb_eval_steps += 1
    print("测试随时间为：",total_test_time)

    all_counts = torch.stack([x[f'span_f1s'] for x in dev_outputs]).sum(0)
    correct_pred, total_pred, total_golden = all_counts
    print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
    precision = correct_pred / (total_pred + 1e-10)
    recall = correct_pred / (total_golden + 1e-10)
    f1 = precision * recall * 2 / (precision + recall + 1e-10)

    res = {
        'span_precision': round(precision.cpu().numpy().tolist(), 5),
        'span_recall': round(recall.cpu().numpy().tolist(), 5),
        'span_f1': round(f1.cpu().numpy().tolist(), 5)
    }
    logger.info("{0} metric is {1}".format(prefix, res))

    # save metrics result
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    per_class = _compute_per_class_span_metrics(dev_outputs, labels)
    split_name = split_tag if split_tag else mode

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {0} {1} *****".format(mode, prefix))
        writer.write("***** Eval results {0} {1} *****\n".format(mode, prefix))
        for key in sorted(res.keys()):
            logger.info("{} = {}".format(key, str(res[key])))
            writer.write("{} = {}\n".format(key, str(res[key])))
        writer.write("per_class_f1:\n")
        for lab in sorted(per_class.keys()):
            m = per_class[lab]
            line = (
                f"  {lab}:  P={m['p']:.5f} R={m['r']:.5f} "
                f"F1={m['f1']:.5f} support={m['support']}\n"
            )
            writer.write(line)

    _append_per_class_history(args, split_name, epoch_num, prefix, res, per_class)

    return res, dev_outputs  # dev_outputs is a list of outputs, which is [outputs1, outputs2 ... ]


def fast_evaluate(args, ckpt_dir, config, tokenizer, labels,
                  mode, prefix='', model=None,only_test=True):
    # if not model:
    #     print("The best dev model was saved in path:",ckpt_dir)
    #     model = BertSpanNerBN.from_pretrained(
    #         ckpt_dir,
    #         config=config,
    #         num_labels=len(labels),
    #         args=args
    #     )
    #     model.to(args.device)


    args.load_path = os.path.join(args.output_dir, "best_model.pth")

    results, predictions = evaluate(
        args,
        model,
        tokenizer,
        labels,
        mode=mode,
        prefix=prefix,
        only_test=only_test,
        epoch_num=getattr(args, "_final_epoch", None),
        split_tag=mode,
    )
    if only_test:
        os.makedirs('saved_weights/2015', exist_ok=True)
        _pk4 = {'pickle_protocol': 4}
        transport_plans=torch.cat(model.transport_plans,dim=0)
        torch.save(transport_plans.cpu().detach().numpy(), 'saved_weights/2015/transport_plans.pt', **_pk4)
        span_index=torch.cat(model.span_indexs,dim=0)
        torch.save(span_index.cpu().detach().numpy(),'saved_weights/2015/span_index.pt', **_pk4)
        span_mask = torch.cat(model.span_mask, dim=0)
        torch.save(span_mask.cpu().detach().numpy(), 'saved_weights/2015/span_mask.pt', **_pk4)
        span_labels = torch.cat(model.span_labels, dim=0)
        torch.save(span_labels.cpu().detach().numpy(), 'saved_weights/2015/span_labels.pt', **_pk4)
        txt_representations=torch.cat(model.txt_representations_IB,dim=0)
        torch.save(txt_representations.cpu().detach().numpy(),'saved_weights/2015/txt_global_distribution_IB.pt', **_pk4)
        txt_representations_ori = torch.cat(model.txt_representations_ori, dim=0)
        torch.save(txt_representations_ori.cpu().detach().numpy(), 'saved_weights/2015/txt_global_distribution_ori.pt', **_pk4)
        img_representations=torch.cat(model.img_representations_IB,dim=0)
        torch.save(img_representations.cpu().detach().numpy(),'saved_weights/2015/img_global_distribution_IB.pt', **_pk4)
        img_representations_ori = torch.cat(model.img_representations_ori, dim=0)
        torch.save(img_representations_ori.cpu().detach().numpy(), 'saved_weights/2015/img_global_distribution_ori.pt', **_pk4)
        txt_representations_OT = torch.cat(model.txt_representations_OT, dim=0)
        torch.save(txt_representations_OT.cpu().detach().numpy(), 'saved_weights/2015/txt_representations_OT.pt', **_pk4)
        img_representations_OT = torch.cat(model.img_representations_OT, dim=0)
        torch.save(img_representations_OT.cpu().detach().numpy(), 'saved_weights/2015/img_representations_OT.pt', **_pk4)
        txt_representations_OT_g = torch.cat(model.txt_representations_OT_g, dim=0)
        torch.save(txt_representations_OT_g.cpu().detach().numpy(), 'saved_weights/2015/txt_representations_OT_g.pt', **_pk4)
        img_representations_OT_g = torch.cat(model.img_representations_OT_g, dim=0)
        torch.save(img_representations_OT_g.cpu().detach().numpy(), 'saved_weights/2015/img_representations_OT_g.pt', **_pk4)
        fusion_text_representations = torch.cat(model.fusion_text_representations, dim=0)
        torch.save(fusion_text_representations.cpu().detach().numpy(), 'saved_weights/2015/fusion_text_representations.pt', **_pk4)



    output_eval_file = os.path.join(ckpt_dir, "{0}_results.txt".format(mode))

    with open(output_eval_file, "a") as writer:
        writer.write('***** Predict in {0} {1} dataset *****\n'.format(mode, prefix))

    return results, predictions


def main():

    # modified, Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    logger.info('The list of lables is: %s ',labels)
    print('Number of labels is: ',len(labels))

    # ------------config--------------
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None
    )
    # ------------tokenizer--------------
    tokenizer_args = {k: v for k, v in vars(args).items()
                      if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    # print(args.tokenizer_name)
    # print(type(args.tokenizer_name))

    #-----for Roberta-------------------
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    )

    # # -----for Bert-------------------
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     cache_dir=None,
    #     **tokenizer_args,
    # )
    # ------------load pre-trained model--------------
    model = BertSpanNerBN(config, args=args,num_labels=len(labels))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('total parameter for the model: {}'.format(total_params))

    # total_para = 0
    # for param in model.parameters():
    #     # print(param.size())
    #     # print(np.prod(param.size()))
    #     total_para += np.prod(param.size())
    # logger.info('total parameter for the model: {}'.format(total_para))
    '''
    #打印模型参数名称及大小
    print("----------"*5)
    for name,p in model.named_parameters():
        print(name,' : ',p.shape)
    '''

    model.to(args.device)
    resume = getattr(args, "resume_checkpoint", "") or ""
    if resume and os.path.isfile(resume):
        try:
            state = torch.load(resume, map_location=args.device, weights_only=False)
        except TypeError:
            state = torch.load(resume, map_location=args.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "resume_checkpoint loaded from %s (missing=%d unexpected=%d)",
            resume,
            len(missing),
            len(unexpected),
        )
    logger.info("Training/evaluation parameters %s", args)
    #print(args.do_train)


    # ------------Training------------
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    best_ckpt_dir = args.output_dir
    # if not os.path.exists(best_ckpt_dir):
    #     os.makedirs(best_ckpt_dir)

    # ------------Evaluation------------
    if args.do_eval:
        # eval best checkpoint
        args.load_path = os.path.join(args.output_dir, "best_model.pth")
        if args.load_path is not None:  # load model from load_path
            logger.info("Loading model from {}".format(args.load_path))
            model.load_state_dict(torch.load(args.load_path))
            logger.info("Load model successful!")
        evaluate(
            args,
            model,
            tokenizer,
            labels,
            mode="dev",
            prefix="best ckpt",
            epoch_num=getattr(args, "_final_epoch", None),
            split_tag="dev",
        )

    # ------------Prediction------------
    if args.do_predict:
        # eval best checkpoint
        args.load_path = os.path.join(args.output_dir, "best_model.pth")
        if args.load_path is not None:  # load model from load_path
            logger.info("Loading model from {}".format(args.load_path))
            model.load_state_dict(torch.load(args.load_path))
            logger.info("Load model successful!")
        results, predictions = fast_evaluate(
            args,
            best_ckpt_dir,
            config,
            tokenizer,
            labels,
            mode="test",
            prefix="best ckpt",
            model=model,
            only_test=True,
        )
        #fitlog.add_metric({"test": {"best_ckpt_f1": results["span_f1"]}}, step=0)

        # Save predictions
        test_file = os.path.join(args.data_dir, "test.txt")
        output_test_predictions = os.path.join(best_ckpt_dir, "test_predictions.txt")
        predictions_save(test_file, predictions, output_test_predictions, labels)


if __name__ == "__main__":
    main()

