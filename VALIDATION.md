# 验证记录

验证日期：2026-10-21。

本次执行环境为 Windows + CPU、Python 3.12.4、PyTorch 2.5.1+cpu、transformers 4.46.3、diffusers 0.31.0；未连接远程服务器，未使用 RTX 4090。正式推荐环境仍是 Ubuntu 22.04、Python 3.10、PyTorch 2.5.1 CUDA 12.1，服务器上需运行 `scripts/check_env.py` 确认。

已执行：

- `python -m pytest -q`：22 项通过。新增覆盖补全基线的观测保留/隐藏目标隔离、未知方法拒绝、多种子汇总与配对差值、检索指标和零向量处理；原有用例继续覆盖掩码、Hungarian ACC、DDIM 复现、条件隔离、图片去重、缓存失效和 Pascal 页面解析。
- `python scripts/smoke_test.py`：通过。使用独立合成特征跑两个种子，覆盖 zero/mean/KNN/ridge/MLP 五个基线、条件与无条件扩散、实验一至七、`metrics_long.csv`、多种子 `summary.csv`/`paired_delta.csv` 和聚合图。
- `python scripts/run_seeds.py --config <synthetic-config> --seeds 42 --exp 1`：通过，验证单种子 CLI、per-seed 输出目录、训练与聚合串联。
- 微型随机 CLIP 测试使用真实 `transformers` CLIPModel/CLIPProcessor，本地保存再读取后执行 `extract_clip.py` 的完整提取逻辑，不下载预训练权重。
- 已在线获取 Pascal Sentences 官方索引并用工程解析器验证：1,000 个唯一图像 ID、20 类，描述数量差异可被接受；未下载并逐一验证全部正式图片。

尚未执行：

- 真实 Pascal Sentences 数据下载、清洗与人工抽查；
- 真实预训练 CLIP 特征提取；
- RTX 4090 上的 5 种子正式训练与评估；
- 课程报告所需的真实指标、图表和结论。

`outputs/smoke/` 的数值仅用于工程自检，不能作为课程实验结果。正式输出以服务器运行产生的 `outputs/seeds/` 和 `outputs/aggregated/` 为准。
