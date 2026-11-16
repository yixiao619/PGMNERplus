# Table 1 数据分析（MINER-My）

## 运行

```bash
cd /home/jyx/yixiao/MINER-My
# 快速自检（约百条内可调小 max_samples）
python analysis/dataset_analysis.py --max_samples 500 --output_dir analysis/results_quick

# 完整（Twitter-2015 + 2017，train/dev/test；耗时取决于 GPU/样本数）
mkdir -p analysis/logs analysis/results
nohup python analysis/dataset_analysis.py \
  --data_root data \
  --output_dir analysis/results \
  --clip_model pretrain/clip-vit-base-patch32 \
  --splits train,dev,test \
  --datasets Twitter-2015,Twitter-2017 \
  > analysis/logs/dataset_analysis.log 2>&1 &
echo $! > analysis/logs/analysis.pid
```

## 图表

分析完成后（`analysis/results/table1_summary.json` 存在）：

```bash
python analysis/make_table1_figures.py
# 输出 analysis/figures/table1_bar_chart.png 等
```

## 说明

- **数据格式**：与 `utils/datasets.load_examples` 一致；图片目录为 `data/twitter2017_images`（即 `data_dir + "_images/"`）。
- **CLIP**：Hugging Face `CLIPModel`（默认本地 `pretrain/clip-vit-base-patch32`），**不是** OpenAI `pip install clip`。
- **COCO-NER / WikiDiverse**：若在 `data/` 下放置对应 `train.json` 及图片目录，可加 `--datasets Twitter-2015,Twitter-2017,COCO-NER,WikiDiverse`；否则会跳过缺失项。
- **与论文数值差异**：阈值（0.25）、patch 网格（3×3）、CLIP 实现细节均会导致与论文 Table 1 **不完全一致**；以本脚本**实测**为准更新论文。
