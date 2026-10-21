# 条件潜在扩散的缺失图文聚类实验报告

> **状态：报告骨架，待真实实验填充。** 当前仓库只在 CPU 合成数据上完成工程自检，本文不包含任何真实课程实验指标。`outputs/smoke/` 的数值不能写入结果章节。

## 摘要

待真实实验完成后填写：数据规模、缺失率、对比方法、主要指标（ACC/NMI/ARI、R@1/R@5、MSE/cosine）和结论。明确说明扩散补全是否优于零填充、均值、KNN、岭回归和 MLP 基线。

## 1. 问题定义

给定成对图文数据 $(I_i, T_i)$，用冻结 CLIP 编码为 $z^I_i, z^T_i \in \mathbb{R}^{512}$。测试时以样本比例 $r$ 随机或定向缺失一个模态，要求在不使用缺失真值、不查看测试标签的前提下补全缺失特征，拼接后做 K-means 聚类，并用 ACC/NMI/ARI 评价。

关键约束：

- 只补缺失槽，观测特征严格保留；
- 模型和基线只在 train 上拟合，超参只在 val 上选择；
- 测试标签只用于最终评价，不进入补全、聚类或调参；
- $r$ 是缺失一个模态的样本比例，不是特征元素比例。

## 2. 方法

### 2.1 冻结 CLIP 特征

使用 `openai/clip-vit-base-patch32` 的图像/文本投影特征，L2 归一化后缓存。CLIP 不微调，特征空间即“潜在空间”。

### 2.2 双向条件扩散补全

两个方向：

- `img2txt`：以图像特征为条件，生成缺失的文本特征；
- `txt2img`：以文本特征为条件，生成缺失的图像特征。

去噪器为两层 512 隐层 MLP，输入为 $[x_t; c; \text{time embedding}]$，预测噪声 $\epsilon$。训练目标为噪声 MSE；推理使用 `diffusers.DDIMScheduler`，余弦调度、epsilon 预测、`eta=0`、trailing 时间步。特征先乘 $\sqrt{512}$，生成后逆缩放并 L2 归一化。

### 2.3 补全与融合

对每个缺失样本，只用观测模态作为条件生成缺失模态。拼接 $[z^I; z^T]$ 后整体 L2 归一化，再做 K-means。实验四额外比较无条件模型和生成侧降权。

### 2.4 无标签基线

| 方法 | 说明 |
|---|---|
| zero | 缺失槽置零，等价于只用观测模态 |
| mean | train 目标模态均值，L2 归一化 |
| KNN | 在观测模态上找 $k$ 近邻，平均目标特征；$k$ 在 val 上选 |
| ridge | 岭回归 source→target；alpha 在 val 上选 |
| MLP | 两层 MLP 回归；按 val 损失早停 |

### 2.5 评价指标

- 补全：缺失位置的 MSE、cosine；
- 聚类：K-means + Hungarian ACC、NMI、ARI；
- 检索：图→文、文→图 R@1/R@5；
- 多种子：mean±std、t 区间、扩散减基线的配对差值。

## 3. 实验设置

> 待真实运行后填写具体数字。

| 项目 | 设置 |
|---|---|
| 数据 | Pascal Sentences，约 1000 图 / 20 类 |
| 划分 | 按类别分层 70%/15%/15%，split seed 42 |
| 设备 | Ubuntu 22.04 + RTX 4090（待运行） |
| CLIP | `openai/clip-vit-base-patch32`，commit 待记录 |
| 扩散 | 100 timesteps，20 DDIM steps，100 epoch，batch 128，lr 1e-3 |
| 种子 | 42–46，共 5 个 |
| 聚类 | K = 真实类别数，`n_init=20`，固定种子 |

待填：有效样本数、每类样本数、划分后 train/val/test 数量、被剔除条目数、运行耗时、GPU 驱动/CUDA/PyTorch 版本。

## 4. 实验结果

> 待真实实验完成后，由 `outputs/aggregated/` 的 CSV 生成表格和图片，不手抄数字。

### 4.1 补全质量（实验一）

待填：两方向、各方法的 MSE/cosine，均值±标准差。

### 4.2 缺失率（实验二）

待填：0/10/30/50/70% 下的 ACC/NMI/ARI 与 R@1/R@5；建议图：三个指标随缺失率变化，每个方法一条线。

### 4.3 缺失模式（实验三）

待填：仅缺图、仅缺文、随机缺失三种模式；建议图：分组柱状图。

### 4.4 消融（实验四）

待填：条件拼接、无条件拼接、生成侧权重 0/.25/.5/.75/1 的聚类指标。

### 4.5 定性可视化（实验五）

待填：观测、扩散补全、val 选出的最佳基线、完整参考四面板联合 t-SNE。只作定性观察。

### 4.6 DDIM 步数（实验六）

待填：5/10/20/50 步的补全与聚类指标。

### 4.7 检索（实验七）

待填：各方法图→文、文→图 R@1/R@5。

## 5. 讨论与局限

待真实结果后填写，至少包括：

- 单数据集、小样本，结论不外推到其他图文数据；
- K 使用真实类别数，属于较强假设；
- 每个条件只生成一次，未分析生成方差；
- 5 个种子的 t 区间为探索性，不足以做强统计显著性声明；
- 扩散补全可能不如 KNN/岭回归，这一负结果同样有价值；
- CLIP 特征是冻结的，补全质量受特征空间结构限制。

## 6. 结论

待真实结果后填写。结论必须与表格一致，不使用 smoke 数值，不根据测试集反复调参后宣称泛化性能。

## 附录 A：复现命令

```bash
# 环境
conda create -n ai-mul python=3.10 -y
conda activate ai-mul
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_env.py

# 数据与特征
python scripts/download_pascal.py --output data
python src/prepare_data.py --config configs/default.yaml
python src/extract_clip.py --config configs/default.yaml

# 多种子训练、实验、聚合
python scripts/run_seeds.py --config configs/default.yaml

# 仅重绘
python src/plot_results.py --config configs/default.yaml
```

## 附录 B：产物说明

```text
outputs/seeds/seed_*/metrics/       # 每个种子的实验 CSV
outputs/seeds/seed_*/checkpoints/   # 每个种子的四个模型
outputs/aggregated/summary.csv      # 多种子 mean/std/CI
outputs/aggregated/paired_delta.csv # 扩散减基线的配对差值
outputs/aggregated/figures/         # 聚合图
```

## 参考文献

1. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
2. Ho et al. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
3. Song et al. Denoising Diffusion Implicit Models. ICLR 2021.
4. 待补充：缺失模态补全、缺失模态聚类、图文检索相关文献。
