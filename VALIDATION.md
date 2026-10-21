# 验证记录

验证日期：2026-10-21。

本次执行环境为 CPU、Python 3.12、PyTorch 2.5.1+cpu；没有访问用户服务器或 RTX 4090。正式推荐环境是 Ubuntu 22.04、Python 3.10、PyTorch 2.5.1 CUDA 12.1，服务器上需运行 scripts/check_env.py 确认。

已执行：

- `python -m pytest -q`：13 项通过。覆盖缺失样本比例及方向、Hungarian ACC、生成侧降权、DDIM 复现和条件信息隔离、非有限/零特征、图片去重及划分隔离、隐藏目标不进入补全、观测模态保留、完整 CLIP 提取与缓存失效检查、官方页面描述数量差异。
- `python scripts/smoke_test.py`：通过。独立合成特征训练两套条件模型和两套无条件模型，运行实验一至五，并生成全部 CSV、总览与图表。
- 微型随机 CLIP 测试使用真实 Transformers CLIPModel/CLIPProcessor，本地保存再读取后执行 extract_clip.py 的完整提取逻辑，不下载预训练权重。
- Python 编译及 bash 语法检查通过。
- 已在线获取 Pascal Sentences 官方索引并用工程解析器验证：1,000 个唯一图像 ID、20 类，996 张有五条描述、3 张有四条、1 张有六条。脚本支持取第一条可用描述。未声称已下载并逐一验证全部正式图片。

尚未执行：真实预训练 CLIP + 完整真实数据集 + RTX 4090 的正式训练与评估；不能根据合成自检推断真实指标、训练速度或最终性能。正式输出以服务器运行产生的 outputs/ 为准。

测试中的 Matplotlib/pyparsing 弃用提示不影响测试通过。自检产物不包含在发布压缩包内，避免将其误当课程结果。
