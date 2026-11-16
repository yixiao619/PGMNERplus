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
import math
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
from utils.datasets import collate_fn, get_labels, load_examples, SpanNerDataset
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
        default='pretrain/clip-vit-base-patch32',
        type=str,
        help="CLIP checkpoint dir or Hugging Face id (e.g. openai/clip-vit-base-patch32)",
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

    args = parser.parse_args()
    # Path to a file containing all labels.
    args.labels = os.path.join(args.data_dir, './labels.txt')
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

    return args


args = get_args()
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


def train(args, model, tokenizer, labels):
    """ Train the model """
    train_examples = load_examples(args.data_dir, mode="train", tokenizer=tokenizer)
    #training_steps = (len(train_examples) - 1 / args.batch_size + 1) * args.epoch
    training_steps=math.ceil(len(train_examples)/args.batch_size)*args.epoch
    # Prepare optimizer and schedule (linear warmup and decay)
    # optimizer, scheduler = prepare_optimizer_scheduler_fusion(args, model, training_steps)
    # optimizer, scheduler = prepare_optimizer_scheduler_complete(args, model, training_steps)
    optimizer, scheduler = prepare_optimizer_scheduler(args, model, training_steps)
    # optimizer, scheduler = prepare_optimizer_scheduler_OT(args, model, training_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", training_steps)

    global_step = 0
    best_score = 0.0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.epoch), desc="Epoch")
    epoch_num = 0
    n_patience=0

    total_training_time=0.0
    for _ in train_iterator:
        epoch_num += 1
        train_dataset = SpanNerDataset(train_examples, args=args, tokenizer=tokenizer, labels=labels)
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

            # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            end_time = time.time()
            train_one_bath_time = end_time - start_time
            total_training_time += train_one_bath_time
            global_step += 1

            if global_step % len(train_dataloader) == 0:
                # default evaluate during training
                # args.load_path = os.path.join(args.output_dir, "best_model.pth")
                results, _ = evaluate(
                    args, model, tokenizer, labels,
                    mode="test", prefix="{}".format(global_step)
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

                    torch.save(model.state_dict(),args.output_dir + "/best_model.pth")
                    # model_save(args, args.output_dir, model, tokenizer)
                    logger.info("Save best model at {}".format(args.output_dir))
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
    avrage_train_time=total_training_time/epoch_num
    print("训练one epoch的平均时间为: ",avrage_train_time)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, mode='test', prefix='', examples=None,only_test=False):

    eval_examples = examples if examples else load_examples(args.data_dir, mode=mode, tokenizer=tokenizer)
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
                'span_label_ltoken': ori_fea_tensors['span_labels']
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

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {0} {1} *****".format(mode, prefix))
        writer.write("***** Eval results {0} {1} *****\n".format(mode, prefix))
        for key in sorted(res.keys()):
            logger.info("{} = {}".format(key, str(res[key])))
            writer.write("{} = {}\n".format(key, str(res[key])))

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

    results, predictions = evaluate(args, model, tokenizer, labels, mode=mode, prefix=prefix,only_test=only_test)
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
        fast_evaluate(args, best_ckpt_dir, config, tokenizer, labels,
                      mode="dev", prefix="best ckpt",model=model,only_test=False)

    # ------------Prediction------------
    if args.do_predict:
        # eval best checkpoint
        args.load_path = os.path.join(args.output_dir, "best_model.pth")
        if args.load_path is not None:  # load model from load_path
            logger.info("Loading model from {}".format(args.load_path))
            model.load_state_dict(torch.load(args.load_path))
            logger.info("Load model successful!")
        results, predictions = fast_evaluate(args, best_ckpt_dir, config,  tokenizer,
                                             labels, mode="test", prefix="best ckpt",model=model,only_test=True)
        #fitlog.add_metric({"test": {"best_ckpt_f1": results["span_f1"]}}, step=0)

        # Save predictions
        test_file = os.path.join(args.data_dir, "test.txt")
        output_test_predictions = os.path.join(best_ckpt_dir, "test_predictions.txt")
        predictions_save(test_file, predictions, output_test_predictions, labels)


if __name__ == "__main__":
    main()

