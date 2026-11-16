# PGMNER-My

PGMNER 的训练与评估脚本入口。模型代码在 [MINER-My](../MINER-My)（`main-mner-robert.py`），本仓库只放实验脚本和配置。

## 环境

```bash
git clone <MINER-My> ../MINER-My    # 与 PGMNER-My 同级目录
cd PGMNER-My
pip install -r requirements.txt     # 在 MINER-My 的 venv 里装也行
```

需要：`pretrain/roberta-large`、`pretrain/clip-vit-base-patch16`，以及 `data/twitter2017`（或 2015）的文本和图片目录。

```bash
bash scripts/check_env.sh twitter2017
bash scripts/prepare_data.sh twitter2017   # 看缺多少图
```

## 主实验

Twitter-2017（默认 dc10_m05：dust=0.8, mass=0.5）：

```bash
GPU=0 SEED=42 bash scripts/train_main.sh twitter2017
```

Twitter-2015：

```bash
GPU=0 SEED=42 bash scripts/train_main.sh twitter2015
```

多 seed：

```bash
GPU=0 bash scripts/train_multiseed.sh twitter2017 42 123 456
```

输出在 `MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed<SEED>/`。

## 评估

```bash
# 读已有 eval_results.txt
python3 scripts/parse_metrics.py ../MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/eval_results.txt

# 重新跑 dev+test（需 best_model.pth）
GPU=0 bash scripts/eval_checkpoint.sh ../MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42

# 只要 test
GPU=0 bash scripts/predict_test.sh ../MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42
```

`parse_metrics.py --per-class` 可打出 PER/LOC/ORG/MISC。

## 消融

矩阵在 `config/ablation_matrix.tsv`。串行跑全部 key：

```bash
GPU=0 bash scripts/train_ablation.sh
```

跑完后汇总：

```bash
python3 scripts/summarize_ablation.py ../MINER-My
```

## 目录

```
config/          主实验与消融超参
scripts/
  lib/           common.sh, hparams.sh
  train_main.sh
  train_multiseed.sh
  train_ablation.sh
  eval_checkpoint.sh
  predict_test.sh
  parse_metrics.py
  summarize_ablation.py
  check_env.sh
  prepare_data.sh
```

超参默认值与 `scripts/lib/hparams.sh`、`config/twitter2017_main.json` 一致；改环境变量或 json 后再训练即可。
