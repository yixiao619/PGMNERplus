# KAIS controlled mechanism evidence

对应 kais1103.tex 第 7.3 节及附录 B。包含原四项机制实验和深化结果，共 M1–M8 八项。

- phenomena/results.json：原四项的完整数值。
- phenomena/kais_deepening/results.json：初始化、代价差距/温度、预算容量、扰动及不变性完整网格。
- phenomena/kais_deepening/BD_results.json：均值归一化门控分散度，以及 84 项门控/温度/质量/代价组合。
- protocol.json、BD_protocol.json：执行前确定的设置。
- runs/：仅包含两组新训练的命令和已冻结的早期探针；不包含 checkpoint 或训练样本。论文固定使用共同的 0–1600 步窗口，不是最后训练结果。
- models/repaired_alignment.py：生成数值实验所用实现。文件名沿用实验仓库；论文中的方法范围由公式说明。
- sha256.json：数据与脚本的内容校验。

M1、M3–M8 是受控数值实验，M2 是真实 TRAIN 早期诊断。无 TEST，无新的任务性能主张。新预算和门控感知代价是明确列出的期刊分析配置；不能把这些结果当作原会议所有配置的性质。

重新绘图：在论文目录运行 python plot_kais1103_mechanisms.py，使用此包已保存的完整数值。脚本需要 numpy、matplotlib。数值重算脚本保留在 phenomena/kais_deepening；它们依赖 torch，并拒绝覆盖已有 results.json，应复制协议和代码至新的输出副本后执行。
