运行前准备
----------
1) 依赖与 GPU 版 PyTorch（尽量用 GPU 跑）
   pip install -r requirements.txt
   再按本机 CUDA 安装带 GPU 的 torch（示例，版本以 https://pytorch.org 为准）：
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   若报错「驱动过旧」，需升级 NVIDIA 驱动，或换与驱动匹配的 cu11x/cu12x 轮子。

   训练默认使用 GPU 0（--gpu_id 0），勿加 --gpu_id -1。

2) 标签文件：data/twitter2017/labels.txt
   本仓库为 Span 级类别：O、LOC、MISC、ORG、PER（与 get_entities 抽取的类型一致，不是 B-/I- 行列表）

3) 预训练模型（二选一）：
   A) 联网：默认使用 Hugging Face 模型名，首次运行会自动缓存
      - 文本：roberta-large 或 roberta-base（见 scripts/run_twitter*_main.sh 与 run_twitter*_roberta_base.sh）
      - 图像：openai/clip-vit-base-patch32（见 main-mner-robert.py --vit_name）
   B) 离线：将权重整库放到 pretrain/ 下，例如：
      - roberta-large：--model_name_or_path pretrain/roberta-large --tokenizer_name pretrain/roberta-large
      - roberta-base ：hf download FacebookAI/roberta-base --local-dir pretrain/roberta-base
      CLIP：--vit_name pretrain/clip-vit-base-patch32

   国内镜像（HF-Mirror）：下载前执行
   export HF_ENDPOINT=https://hf-mirror.com
   详见 https://hf-mirror.com

   命令行工具：新版包提供的是 hf（若提示 command not found，先 pip install -U "huggingface_hub[cli]"，
   并确认用的是装包的那个 Python 环境；或用下面的 python -m，不依赖 PATH）。

   ---------- 全量下载 RoBERTa-large（与网页仓库一致，约 7GB+）----------
   在项目根目录 MINER-My 下执行（可先删掉旧的 pretrain/roberta-large 再下）：

   export HF_ENDPOINT=https://hf-mirror.com
   pip install -U "huggingface_hub[cli]"
   hf download FacebookAI/roberta-large --local-dir pretrain/roberta-large

   若没有 hf 命令，用：
   python3 -m huggingface_hub.cli.hf download FacebookAI/roberta-large --local-dir pretrain/roberta-large

   （等价「克隆整库」：用 Git LFS，同样走镜像）
   cd pretrain && rm -rf roberta-large
   git lfs install
   GIT_LFS_SKIP_SMUDGE=0 git clone https://hf-mirror.com/FacebookAI/roberta-large roberta-large
   cd ..

   代码默认 --model_name_or_path / --tokenizer_name 为 pretrain/roberta-large，无需改参数。

   CLIP（视觉分支，与 RoBERTa 分开下）：
   export HF_ENDPOINT=https://hf-mirror.com
   hf download openai/clip-vit-base-patch32 --local-dir pretrain/clip-vit-base-patch32
   # 或：python3 -m huggingface_hub.cli.hf download openai/clip-vit-base-patch32 --local-dir pretrain/clip-vit-base-patch32

4) 运行（默认 GPU 0 + 本地 pretrain 路径）：
   cd /path/to/MINER-My
   python main-mner-robert.py --gpu_id 0

   指定其它卡：python main-mner-robert.py --gpu_id 1

   无 GPU：python main-mner-robert.py --gpu_id -1
   仅训练不测：python main-mner-robert.py --do_predict false
   布尔参数请用 true/false（已修复，勿依赖 type=bool 的旧行为）

参考：roberta-large 页面 https://huggingface.co/FacebookAI/roberta-large

论文式训练命令与参数对照（含结果记录、Twitter2015 数据说明、消融）：见 scripts/EXPERIMENTS.txt
一键主实验：Twitter2017 → bash scripts/run_twitter2017_main.sh
Twitter2015（先按 data/twitter2015/README.txt 放数据）→ bash scripts/run_twitter2015_main.sh
给 Cursor 的一页操作说明：scripts/CURSOR_TWITTER2015_INSTRUCTIONS.txt
