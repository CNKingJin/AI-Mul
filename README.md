# AI-Mul：条件潜在扩散的缺失图文聚类

面向 Ubuntu 22.04 + RTX 4090 的课程实验工程。实现路线：**冻结 CLIP → 双向条件特征扩散 → 缺失模态补全 → 特征拼接 → K-means → 多组实验**。不生成图片或文本，不微调 CLIP。当前版本加入零填充、训练集均值、KNN、岭回归、MLP 五个无标签补全基线，并在原五组实验上增加多种子聚合、DDIM 步数敏感性和跨模态检索。

> **当前状态（2026-10-21）**：代码已在 Windows CPU + Python 3.12 + PyTorch 2.5.1+cpu 上完成 `pytest` 与合成数据 `smoke_test`；真实 Pascal 数据、预训练 CLIP、RTX 4090 正式训练和课程报告结果尚未运行。smoke 数值只用于工程自检，不能当作实验结果。

这里的“潜在”指 CLIP 投影后的特征空间，没有另训 VAE。代码是根据实验路线实现的轻量工程，**不是某篇论文官方复现，也不保证扩散提升聚类表现**。扩散采样复用官方 `diffusers.DDIMScheduler`，其余任务逻辑和基线实现在本仓库。

如果你拿到的是压缩包：先将压缩包放到服务器，执行 `unzip AI-Mul-experiments.zip`，然后 `cd AI-Mul`，从下面的环境配置继续。只有代码已经上传仓库后才执行 `git clone`。

## 1. 获取代码与配置环境

```bash
git clone https://github.com/CNKingJin/AI-Mul.git
cd AI-Mul
# 当前 main 分支已包含实现代码
nvidia-smi
```

推荐 Python 3.10（Ubuntu 22.04 默认版本），以下二选一：

```bash
# 已安装 Conda 的服务器
conda create -n ai-mul python=3.10 -y
conda activate ai-mul
```

```bash
# 不使用 Conda：需系统已安装 python3-venv
python3 -m venv .venv
source .venv/bin/activate
```

随后安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/check_env.py
```

这里只需 PyTorch CUDA wheel 和兼容 NVIDIA 驱动，无需另装 CUDA Toolkit。该 wheel 使用 CUDA 12.1；若驱动过旧或已有可用 PyTorch，请先参考 [PyTorch 官方安装历史](https://pytorch.org/get-started/previous-versions/)选兼容版本。驱动兼容性以 PyTorch/CUDA 官方说明和环境检查结果为准。不要为了该项目直接升级共享服务器驱动。

多 GPU 服务器使用下面方式只暴露一张卡，卡号按实际空闲情况修改：

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 2. 先做代码自检

```bash
python -m pytest -q
python scripts/smoke_test.py
```

自检会在 `outputs/smoke/` 里生成独立合成数据，为两个种子各训练四个小模型，运行实验一至七（含五个补全基线）、聚合并绘图；它不下载数据或预训练权重，也不改变正式配置。**这些数值只用于检查工程，不能作为课程实验结果。** 测试还使用一个本地随机初始化的微型 CLIP 检查完整特征提取接口；这不等于验证真实预训练模型效果。

## 3. 准备真实图文数据

### 方式 A：从零开始，用官方 Pascal Sentences

实验路线允许 Wikipedia 获取困难时改用 Pascal Sentences。本工程提供官方来源下载脚本，省去手工整理：

```bash
python scripts/download_pascal.py --output data
```

脚本解析 [UIUC 官方页面](https://vision.cs.uiuc.edu/pascal-sentences/)，读取图片 URL 中的原始类别目录，下载 1,000 张图片，固定取每图第一条可用描述（多数有五条，官方个别条目数量不同），生成 `data/manifest.csv`。原始页面、全部描述也会保存。下载会验证图片，支持重新运行跳过完整图片；任何条目失败即停止，不能静默把不完整数据当成完整数据集。网站不可达时，可在联网电脑执行后复制整个 `data/` 目录到服务器。

类别采用来源目录的单类别划分，不是完整 VOC 多标签检测任务。后续按实际清洗数量记录，不预填数据集结果。描述按原始授权使用，详见 [data/README.md](data/README.md)。

### 方式 B：已有 Wikipedia 或其他合规配对数据

整理成 UTF-8 CSV，修改 `configs/default.yaml` 的 `manifest` 路径：

```csv
sample_id,image_path,text,label
000001,images/000001.jpg,"A dog playing on grass",dog
000002,images/000002.jpg,"A red bus on a road",bus
```

要求：

- 图片路径相对 **manifest.csv 所在目录**，也接受绝对路径。
- 一图一条描述、唯一 ID、真实单类别标签。不要从描述自动推断“真实标签”。
- 不同划分不能有相同图片；程序按图片内容 SHA256 去重，记录被剔除条目。
- 至少两个类别，每类至少七张有效不同图片；分层划分还要求 val/test 容量足以覆盖全部类别，小数据应增加样本。
- 人工抽查图文配对；程序无法判断一句描述是否真的对应图片。

运行准备并检查统计：

```bash
python src/prepare_data.py --config configs/default.yaml
```

输出 `data/splits/{train,val,test}.csv`、`metadata.json` 和 `rejected.csv`。按类别分层划分为约 70%/15%/15%，种子 42；标签只用于分层、确定类别总数和最终评价，不进入扩散网络或 K-means 输入。

## 4. 准备 CLIP 并提取特征

默认在线加载 `openai/clip-vit-base-patch32`：

```bash
python src/extract_clip.py --config configs/default.yaml
```

每个集合缓存：

```text
features/train/z_img.npy      # N × 512，单位向量
features/train/z_txt.npy      # N × 512，单位向量
features/train/ids.csv        # 仅 sample_id，不含标签
features/train/provenance.json
# val 和 test 同样结构
```

缓存记录 split 文件摘要、模型名称和实际模型 commit。正式复现可把该 commit 写入 `clip.revision`。修改数据、模型或划分后重新提取特征并重新训练；程序会检查 split 摘要、ID 顺序和 checkpoint 对应的 train/val 特征摘要，避免误用旧文件。`configs/default.yaml` 中 `clip.batch_size` 可降到 16 解决特征提取显存不足。

### 服务器离线时

在联网机器安装依赖，下载完整权重及处理器文件：

```bash
python scripts/download_clip.py --output models/clip-vit-base-patch32
```

将整个模型目录复制到服务器，把 YAML 改为：

```yaml
clip:
  model: models/clip-vit-base-patch32
  revision: null
  local_files_only: true
  batch_size: 32
```

保留配置中的其他字段，再执行特征提取命令。数据下载和 pip 依赖也需要预先准备。若需依赖离线包，请在相同 Python 版本、Linux x86_64 的联网环境用 `pip download` 分别下载 CUDA PyTorch wheel 和 `requirements.txt` 依赖，复制后用 `pip install --no-index --find-links <wheel目录>` 安装。

## 5. 训练模型

```bash
# 两套条件模型：图像→文本、文本→图像
python src/train_diffusion.py --config configs/default.yaml --direction both

# 实验四专用：去掉条件信息的两套模型
python src/train_diffusion.py --config configs/default.yaml --direction both --unconditional
```

也可用 `--direction img2txt` 或 `--direction txt2img` 单独训练。无条件版本保持相同网络参数量，条件输入全部置零，训练与推理一致；这是控制条件信息的消融。传统补全基线由 `src/baselines.py` 单独实现，只使用 train 拟合、val 选超参。

正式多种子流程会为 `configs/default.yaml` 中的每个种子训练四个模型，并自动聚合：

```bash
python scripts/run_seeds.py --config configs/default.yaml
# 只跑一个种子：
python scripts/run_seeds.py --config configs/default.yaml --seeds 42
# 已有 seeds/ 结果时只重新聚合：
python src/aggregate.py --config configs/default.yaml
```

默认配置：512 维 CLIP 特征、128 维正弦时间嵌入、两层 512 隐层 MLP、100 个训练噪声步、20 步确定性 DDIM、100 epoch、batch 128、AdamW lr=1e-3。每个 epoch 使用同一份验证噪声和时间步；按最低验证噪声 MSE 保存最佳权重。测试集合不用于训练、选模型或确定超参数。基线只在 train 上 `fit`，超参只在 val 上选择。

相对路线的两处数值处理：

1. 默认改用余弦噪声调度：100 步线性 beta 的终点仍保留较多信号，直接从标准高斯采样不匹配；训练会拒绝终点 alpha_bar >0.01 的配置。DDIM 使用 trailing 时间步，从训练末端开始。
2. L2 特征的每维数值很小，训练前将目标和条件同时乘 sqrt(512)，默认 22.6274；生成后逆缩放再做 L2 归一化。不引入标签或测试集统计。改变维度时同步修改 `train.feature_scale` 为该维度平方根。

结果保存在 `outputs/checkpoints/`，损失 CSV 与配置、依赖、设备信息在 `outputs/logs/`。训练轮数是起始参数，检查损失曲线，不保证指定轮数收敛。当前训练脚本重新运行会从头训练，不提供断点续训。

## 6. 执行五组实验

```bash
python src/run_experiments.py --config configs/default.yaml --exp all
python src/plot_results.py --config configs/default.yaml
```

也支持把 `all` 换为 `1` 到 `5` 分别执行；实验四需要无条件模型。不同缺失率共享已训练模型，无需重新训练。

| 实验 | 实际内容 | 主要结果文件（outputs/metrics/） |
|---|---|---|
| 一 | 两方向分别遮蔽 30%，只在缺失位置计算 MSE、余弦相似度；5 个基线 + 扩散 | exp1_reconstruction.csv、exp1_per_sample.csv |
| 二 | 缺失率 10%、30%、50%、70%，另有 0% 完整输入参考；5 个基线 + 扩散 | exp2_missing_rate.csv |
| 三 | 50% 样本仅缺图像、仅缺文本、双向随机缺失；5 个基线 + 扩散 | exp3_missing_pattern.csv |
| 四 | 同一 50% mask：条件拼接、无条件拼接、生成侧权重 0/.25/.5/.75/1 | exp4_ablation.csv |
| 五 | 观测、扩散补全、val 选出的最佳基线、完整参考表示联合 PCA + t-SNE | exp5_tsne_coordinates.csv |
| 六 | 同一 50% mask 下 DDIM 步数 5/10/20/50 的敏感性 | exp6_ddim_steps.csv |
| 七 | 50% 缺失下的跨模态检索，图→文、文→图 R@1/R@5 | exp7_retrieval.csv |

`r` 表示缺失一个模态的**样本比例**，不是特征元素比例。任何样本至少保留一个模态。随机缺失方向尽量各半；同种子下缺失样本集合随缺失率增大而嵌套。mask、预测特征、簇编号和样本 ID 分别保存在 `outputs/masks/` 与 `outputs/predictions/`，便于核查和重新计算。

基线包括 zero（缺失槽置零，等价于只用观测模态）、mean（train 目标模态均值）、KNN、岭回归和 MLP；它们只在 train 上拟合，超参只在 val 的缺失 mask 上选择，测试集不参与。所有补全结果与扩散生成特征一样做 L2 归一化。

主方法拼接 `[z_img; z_txt]` 后整体归一化；只补缺失槽，观测特征严格保留。K 使用已知数据类别总数，K-means `n_init=20`、固定种子；不根据测试标签挑选 K 或挑最好结果。ACC 用 Hungarian 对齐，NMI/ARI 用 sklearn，保存为小数；**ARI 可能为负**，不做截断。每个种子还输出 R@1/R@5、补全 MSE/cosine 和 `metrics_long.csv`，供聚合脚本使用。

实验五的置零观测表示、完整特征仅用于可视化/完整输入参考，不作为额外补全方法。四种表示合并后拟合一个 PCA 和一个 t-SNE，再拆成四个共享坐标面板；最佳基线由 val 补全 MSE 选出，不使用测试标签。t-SNE 只供定性观察，不能据此证明提升。

## 7. 一键执行及产物位置

完成依赖、下载数据后：

```bash
bash scripts/run_all.sh configs/default.yaml
```

该命令会重新执行划分、提取，然后按 `configs/default.yaml` 的 `seeds` 逐个种子训练四个模型、运行实验一至七并聚合绘图；中途失败即停止。已完成提取/训练时，使用上面的分步命令跳过已有阶段。

```text
configs/default.yaml          # 全部主要参数
src/prepare_data.py           # 清洗与划分
src/extract_clip.py           # 冻结 CLIP 特征
src/diffusion.py              # 模型、噪声调度、DDIM 采样
src/train_diffusion.py        # 训练与验证选优
src/baselines.py              # zero/mean/KNN/ridge/MLP 基线
src/evaluation.py             # mask、融合、聚类、检索和指标
src/run_experiments.py        # 实验一至七
scripts/run_seeds.py          # 多种子训练、实验、聚合编排
src/aggregate.py              # 多种子 summary/paired delta
src/plot_results.py           # 从 CSV 重绘
outputs/seeds/seed_*/checkpoints/  # 每个种子的四个最佳模型
outputs/seeds/seed_*/logs/         # loss、环境、评估配置、权重摘要
outputs/seeds/seed_*/metrics/      # 每个种子的实验 CSV 与 metrics_long.csv
outputs/seeds/seed_*/figures/      # 每个种子的图
outputs/aggregated/summary.csv     # 多种子 mean/std/CI
outputs/aggregated/paired_delta.csv # 扩散减各基线的配对差值
outputs/aggregated/figures/        # 聚合图
```

只需重绘时运行 `plot_results.py`，无需训练。模型、数据和实验输出都被 `.gitignore` 排除，避免将大文件或数据授权内容推入 Git。课程报告骨架见 `REPORT.md`，真实结果待服务器运行后填充。

## 8. 常见问题与报告边界

- **CUDA 不可用**：先看 `nvidia-smi`，再运行环境检查；确认安装的 torch 不是 CPU wheel，且激活了正确环境。自检使用 CPU，不代表服务器 GPU 已验证。
- **模型下载失败**：使用上面的离线模型目录。不要只复制权重而漏掉 tokenizer、processor 和 config。
- **划分报错**：检查每类有效样本数、总样本数是否能支持 val/test 覆盖所有类别，查看 rejected.csv；不要为了通过划分直接删掉难分类类别而不记录。
- **旧缓存/旧权重报错**：数据改变后重新提取并训练；不要绕过校验。更换任务时建议修改 features_dir、splits_dir、output_dir 使用独立目录。
- **训练损失下降但补全差**：噪声预测损失不等价于语义质量，查看验证曲线、生成余弦和实际聚类结果，不根据测试结果反复调参后宣称泛化性能。
- **结果不提升**：如实报告。小样本随机生成可能削弱类别信息；简单基线（KNN、岭回归）在特征空间里可能强于扩散补全，这是正常且值得报告的结论。5 个种子和 t 区间仍属探索性，不保证不同设备结果逐位一致。
- **指标单位**：ACC/NMI 通常 0–1，ARI 可负；MSE 为归一化特征每维均方误差。不要直接把数值作为百分数绘图。

实验路线默认使用 `configs/default.yaml` 中的 5 个种子；报告应注明具体数据版本、有效对数、类别数、划分、设备、模型 commit、种子集合，并说明 t 区间在少量种子下仅为探索性、未做大规模统计显著性检验。当前仓库只完成 CPU 合成数据自检，不附造出的真实实验结果。

## 9. 实现依据与复用范围

- [CLIP 原始实现](https://github.com/openai/CLIP) 与 [Transformers CLIP 接口](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/clip)：使用安装的 transformers 库加载官方预训练特征，不复制第三方项目代码。
- [Diffusers DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim) 与 [DDIM 论文](https://arxiv.org/abs/2010.02502)：直接调用库的 add_noise/step，epsilon 预测、eta=0、余弦调度。
- [Pascal Sentences 官方数据](https://vision.cs.uiuc.edu/pascal-sentences/)：自写页面解析与下载逻辑，保留来源及描述；没有复制第三方下载器。

补全基线使用 scikit-learn 的 `Ridge`、`NearestNeighbors` 和 PyTorch MLP；数据处理、缺失掩码、扩散训练、实验编排、聚合和测试代码为本工程实现。依赖及原始模型/数据遵循各自许可。

## 10. 将压缩包代码上传到你的空仓库

在你已配置 GitHub 写权限的机器上，进入解压后的 AI-Mul 目录：

```bash
git init -b main
git add .
git commit -m "Add baselines, multi-seed experiments and CPU validation"
git remote add origin https://github.com/CNKingJin/AI-Mul.git
git push -u origin main
```

若你已经克隆仓库，将压缩包内的文件复制到那个仓库目录，然后执行 `git add .`、`git commit`、`git push` 即可，不要重复添加 origin。若远端已有其他提交，先正常拉取合并，不使用强制推送。身份认证使用你本机的 SSH 或 GitHub 凭据管理器，不要把 token 写入代码或 README。
