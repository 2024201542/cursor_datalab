# CrossCheck：多模型互检分类

让多个大模型像多名标注员一样，对同一批文本**独立分类 → 一致即采纳 → 分歧交人工（可选交叉复核 / 仲裁）**，用模型之间的相互检查来提升分类准确率，并把真正困难的样本筛出来交给人。

提示词里会自动放入训练集中最相似的已标注样本（动态示例），还可以加入一个用训练集训练的本地小模型作为额外一票，让投票者的出错方式更独立。

> 当前版本是 **v0.1 基础实现**，已接入真实模型并完成实测；进阶能力见下方 [项目规划](#项目规划roadmap)。
>
> 图文版项目介绍（思路、流程、两次检验结果、路线图）：[`docs/index.html`](docs/index.html)，单文件、无外部依赖，下载后用浏览器直接打开。

---

## 为什么要互检

- **不同模型的分类标准和准确率不同**：同一条文本，GPT、Claude、DeepSeek、Qwen 可能给出不同答案。
- **单个模型的错误难以发现**：模型答错时往往依然“很自信”。
- **不同厂商模型的错误相对独立**：多个模型同时犯同一个错的概率远低于单个模型犯错的概率，因此“一致”本身就是一个很强的质量信号，“分歧”则是一个天然的难例探测器。

互检的本质是把众包标注领域成熟的方法（多数投票、交叉审核、仲裁、Dawid-Skene 聚合、人工复核）迁移到大模型上。

---

## 工作流程

```mermaid
flowchart TD
    A[待分类文本] --> R[检索训练集中最相似的已标注样本<br/>作为动态示例放进提示词（可选）]
    R --> B[多个不同厂商模型 + 本地小模型（可选）并行独立分类<br/>输出 标签 + 置信度 + 理由]
    B --> C{有效结果全部一致?}
    C -- 是 --> D[首轮一致通过]
    C -- 否 --> P{disagreement_action}
    P -- human（默认） --> K
    P -- arbiter --> H
    P -- review --> E[交叉复核<br/>每个模型看到其他人的匿名意见后重新判断]
    E --> F{过半同意 且 加权得票占比 >= 阈值?}
    F -- 是 --> G[复核后多数通过]
    F -- 否 --> H[仲裁模型给出最终判断]
    H --> I{仲裁置信度 >= 阈值?}
    I -- 是 --> J[仲裁模型决定]
    I -- 否 --> K[需人工审核<br/>附带各模型理由和建议标签]
```

几个关键设计：

| 设计 | 目的 |
|---|---|
| 统一的分类手册（定义 + 正反例 + 边界规则） | 让所有模型按同一套标准判断，减少“标准不一致”造成的分歧 |
| 动态示例：每条文本检索训练集中最相似的 k 条已标注样本 | 让模型学到该数据集自己的标注尺度，而不是按常识判断 |
| 本地小模型（TF-IDF + 逻辑回归）作为额外一票 | 与大模型出错方式不同，“全票一致”更可信 |
| 复核时隐去模型名、要求“理由更符合标准才修改” | 防止模型盲从多数或迎合强模型（sycophancy） |
| 加权投票（权重 × 置信度） | 准确率高的模型话语权更大，权重可由金标准自动计算 |
| 回复缓存 | 相同模型 + 相同提示词只调用一次，反复实验不重复付费 |
| mock 模式 | 没有 API key 也能跑通全流程 |

---

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

Python 3.10+。

### 2. 网页平台（推荐）

```bash
streamlit run app.py
```

浏览器打开 <http://localhost:8501>，五个页面：

| 页面 | 功能 |
|---|---|
| ① 模型配置 | 表格编辑模型（投票 / 仲裁、接口地址、模型名、限流、额外参数），填写 API Key（保存到本地 `.env`），一键测试连通性 |
| ② 分类任务 | 编辑类别定义、正反例、边界规则，预览发给模型的提示词 |
| ③ 训练数据（可选） | 没有训练数据可以跳过。有已标注的同类数据时，上传文件或手动录入（追加 / 替换），开启“动态示例”和“本地小模型”；可输入一段文本试看检索结果 |
| ④ 运行与结果 | 上传 CSV / Excel / JSONL（或选 `data/` 目录里的文件），选择文本列和标签列，设置分歧策略、阈值和级联模型；运行前可预估调用次数和费用，运行后显示各模型 token 用量与实际费用 |
| ⑤ 人工审核 | 审核任意一次运行的结果：逐条审核或表格批量审核，查看审核统计，导出最终结果，把审核结果回流为金标准或训练数据 |

**训练数据页**

- **有没有训练数据是两种模式**：没有时，模型只按分类任务页的定义和规则判断，两个增强开关不可用；有时才能开启
- 训练数据保存为 `text,label` 两列的 CSV，默认放在 `data/train/`（已被 git 忽略，不会上传）；标签不在当前类别中的行会被跳过并提示
- 显示样本数、各类别样本量，提示缺样本或样本过少（< 30 条）的类别
- 评估时如果待评估数据和训练数据有重叠，运行页会提醒：本地小模型见过这些答案，准确率会偏高
- 人工审核页可以把审核过的样本直接追加进训练数据文件，越用越准

- **有标签列**：显示每个模型、多数投票、加权投票、互检系统的准确率卡片和对比表，5 张图表，逐条明细（判错标红）
- **无标签列**：显示模型两两一致率、各模型的类别分布、逐条结果
- 可下载结果 CSV、待人工审核 CSV、完整明细 JSONL、图表报告 HTML（单文件，图片已内嵌）
- 侧边栏可切换 / 保存配置文件

**人工审核页**

- **审核范围**：需人工审核的样本 / 首轮有分歧的全部样本 / 从首轮一致通过的样本中随机抽检 N 条 / 全部样本
- **逐条审核**：显示原文、各模型首轮与复核后的判断和理由、仲裁意见；每个类别标注得票数和“模型建议”；确认后立即存盘（`<运行目录>/<结果文件名>_reviews.json`），关闭浏览器不丢失
- **表格批量审核**：表格里直接选人工标签；“未填写的全部采用模型建议”适合抽检场景
- **审核统计**：在已审核样本上，互检系统和每个模型与人工判定的一致率，按处理环节的确认率
- **导出与回流**：下载合并人工结果的最终结果 CSV；把已审核样本追加到金标准文件（按文本去重），下次评估直接使用

### 3. 命令行：用 mock 模式跑通流程（不需要 API key）

```bash
python -m crosscheck classify data/sample.csv --mock
python -m crosscheck evaluate data/gold.csv --mock
```

### 4. 接入真实模型

在项目根目录新建 `.env` 文件写入 key（已被 `.gitignore` 忽略，不会上传），程序启动时自动加载：

```ini
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...      # 阿里云百炼：千问、Kimi、GLM 都可以通过它调用
MOONSHOT_API_KEY=sk-...       # Kimi 官方（可选）
```

```bash
python -m crosscheck ping      # 逐个测试模型是否可用
```

当前默认配置：投票模型为 `deepseek-flash`（DeepSeek 官方）、`kimi-k2.6`（百炼）、`qwen3.7-plus`（百炼），仲裁模型为 `glm-5.3`（百炼，第四家厂商，不偏向任何投票方）。

接入时的常见坑（已在 `config.yaml` 中处理）：

- 新一代模型多数默认开启“思考模式”，分类任务建议通过 `extra_body` 关闭以提速（DeepSeek：`thinking: {type: disabled}`；百炼：`enable_thinking: false`）
- Kimi 官方接口不允许 `temperature=0`（思考模式只允许 1，非思考只允许 0.6）
- 新注册账号限额很低（例如 Kimi 官方并发 1、每分钟 3 次），可用 `max_concurrency` / `rpm` 限流

`provider` 支持三种：

| provider | 适用 |
|---|---|
| `openai` | 所有 OpenAI 兼容接口：OpenAI、DeepSeek、通义千问、Kimi、智谱、大部分中转平台 |
| `anthropic` | Anthropic 原生 `/v1/messages` 接口 |
| `mock` | 假模型，用于测试 |

### 5. 命令行推荐的使用顺序

```bash
# ① 在人工标注的金标准上评估，生成模型权重 output/weights.json
python -m crosscheck evaluate data/gold.csv

# ② 查看 output/eval_report.txt，根据判错样本修改 config.yaml 中的分类标准和边界规则，重复 ①

# ③ 正式分类（先用 --limit 试跑）
python -m crosscheck classify data/your_data.csv --weights output/weights.json --limit 50
python -m crosscheck classify data/your_data.csv --weights output/weights.json

# ④ （可选）无监督交叉验证：Dawid-Skene 估计每个模型的准确率
python -m crosscheck aggregate output/results.jsonl
```

---

## 命令说明

| 命令 | 作用 | 主要输出 |
|---|---|---|
| `classify <输入文件>` | 互检分类 | `results.csv`、`results.jsonl`、`results_need_human.csv` |
| `evaluate <金标准文件>` | 评估系统和每个模型的准确率，自动算权重 | `eval_report.txt`、`weights.json` |
| `aggregate <results.jsonl>` | Dawid-Skene 无监督聚合，无需标准答案即可估计各模型准确率 | `ds_results.csv` |
| `ping` | 测试所有模型 API 连通性 | 控制台输出 |

通用参数：`-c` 配置文件、`-o` 输出目录、`--mock`、`--text-col` / `--id-col` / `--label-col` 指定列名、`--weights` 加载权重、`--limit` 只处理前 N 条。

成本相关参数（`classify` / `evaluate`）：`--estimate-only` 只预估调用次数和费用、`--budget N` 预计费用超过 N 元时不运行、`--disagree-rate` 预估用的分歧比例（默认 0.25）、`--resume` 断点续跑。

输入支持 `.csv`（UTF-8 / GBK 自动识别）和 `.jsonl`。

### 输出文件

- `results.csv`：每条样本的最终标签、处理状态、置信度，以及每个模型每一轮的判断，可直接用 Excel 打开。
- `results_need_human.csv`：需要人工处理的样本，附带建议标签和所有模型的理由，`human_label` 列留给人工填写。
- `results.jsonl`：完整明细（含模型原始回复），用于追溯和二次分析。

### 评估报告与图表

`evaluate` 会在输出目录生成 `report.html`（浏览器打开）和 `charts/` 下的 5 张图：

| 图 | 回答的问题 |
|---|---|
| `accuracy.png` | 每个模型单独的准确率 vs 多数投票 vs 加权投票 vs 完整互检，以及最佳单模型和理论上限参考线 |
| `per_class.png` | 每个类别上哪个模型更强 |
| `review.png` | 交叉复核前后每个模型的准确率变化 |
| `status.png` | 样本在各环节（首轮一致 / 复核通过 / 仲裁 / 人工）的去向和各环节准确率 |
| `confusion.png` | 互检系统的混淆矩阵 |

报告中还有逐条明细表，判错的格子标红。

### 评估指标

- **单模型准确率**（首轮 / 复核后）：衡量每个模型的能力，以及交叉复核是否带来提升。
- **自动采纳比例 & 自动采纳准确率**：系统能替代多少人工，以及替代部分有多可靠。
- **按状态分组的准确率**：通常“首轮一致通过”最可靠，可据此决定哪些状态需要抽检。
- **Fleiss' Kappa**：模型间一致性。过低通常说明分类标准本身有歧义。
- **混淆矩阵**：哪两个类别最容易混淆，是修订边界规则的直接依据。

---

## 实测结果（2026-09）

投票模型 DeepSeek / Kimi / 千问，仲裁模型 GLM。

**客服消息 5 分类**（`data/gold.csv`，60 条，自编，边界规则清晰）

```bash
python -m crosscheck evaluate data/gold.csv -o output/service
```

三个模型均为 100%，Kappa = 1.0。说明：**标准清晰、任务简单时，现代大模型之间几乎没有分歧，互检只起确认作用**。

**新闻标题 15 分类**（`data/tnews_gold.csv`，CLUE TNEWS 验证集分层抽样 150 条）

```bash
python -m crosscheck evaluate data/tnews_gold.csv -c configs/tnews.yaml -o output/tnews
```

| 方案 | 准确率 |
|---|---|
| deepseek-flash | 56.7% |
| kimi-k2.6 | 55.3% |
| qwen3.7-plus | 58.0% |
| 多数投票 | 55.3% |
| 加权投票 | 55.3% |
| 互检系统（复核 + 仲裁） | 57.3% |
| 理论上限（任一模型答对） | 63.3% |

关键发现：

1. **模型错误高度相关**（Kappa = 0.85），投票能提升的空间只有理论上限与最佳单模型之间的 5 个百分点。TNEWS 标签来自头条频道，本身噪声较大，也压低了上限。
2. **“首轮是否一致”是极强的质量信号**：首轮三家一致的 119 条准确率 64.7%，有分歧的 31 条只有 29%。
3. **交叉复核存在“从众收敛”**：分歧样本复核后多数变成三家一致，但一致后的准确率仍只有 24%。因此对这类任务，**首轮分歧本身就应该触发人工或仲裁**，而不是看复核后是否一致。

> TNEWS 数据来自 [CLUE Benchmark](https://github.com/CLUEbenchmark/CLUE)。

### 经济 / 金融文本分类（3 个公开数据集）

```bash
python scripts/prepare_econ.py      # 下载数据，每类分层抽 50 条测试集，训练集短样本作为提示词示例
python -m crosscheck evaluate data/econ_climate_gold.csv -c configs/econ_climate.yaml -o output/econ_climate
python -m crosscheck evaluate data/econ_fomc_gold.csv    -c configs/econ_fomc.yaml    -o output/econ_fomc
python -m crosscheck evaluate data/econ_finfe_gold.csv   -c configs/econ_finfe.yaml   -o output/econ_finfe
```

| 任务（经济学领域） | 数据集 | deepseek | kimi | qwen | 多数投票 | 互检系统 | 理论上限 | 首轮一致 / 分歧的准确率 |
|---|---|---|---|---|---|---|---|---|
| 年报气候段落：风险/机遇/中性（ESG、气候金融） | [ClimateBERT climate_sentiment](https://huggingface.co/datasets/climatebert/climate_sentiment) | 79.3% | 83.3% | **86.0%** | 84.7% | 83.3% | 92.0% | 91.5% / 53.1% |
| FOMC 句子：鹰派/鸽派/中性（货币经济学） | [Trillion Dollar Words](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | **65.3%** | 62.0% | 60.7% | 61.3% | 62.0% | 73.3% | 69.6% / 37.1% |
| 股吧帖子：积极/消极/中性（行为金融、投资者情绪） | [BBT-FinCUGE FinFE](https://github.com/supersymmetry-technologies/BBT-FinCUGE-Applications) | **70.0%** | 68.0% | 67.3% | 68.0% | 68.7% | 79.3% | 75.2% / 45.5% |

结论与 TNEWS 一致：投票不稳定地超过最佳单模型（Kappa 0.76–0.79，错误高度相关），但**首轮一致的约 78% 样本准确率明显更高，分歧的约 22% 是人工复核最该看的部分**。FOMC 的主要错误是把“描述经济强劲”的鹰派句判为中性，这类数据集的标注规则（见 `configs/econ_fomc.yaml`）比常识更严格；FinFE 中有不少反讽和标注噪声（如“今天能涨停算我输”被标为中性）。

#### 改进实验：动态示例 + 本地小模型 + 分歧转人工

上表暴露的问题：三个大模型错误高度相关，交叉复核改对 13 条、改错 10 条（600 条金标准中的 131 条分歧样本），几乎没有净收益。为此做了三项改进，并在同样的 450 条金标准上对比：

1. **动态示例**（`fewshot`）：对每条文本，用字符级 TF-IDF 从训练集检索最相似的 6 条已标注样本放进提示词。训练集已剔除与测试集重复的文本。
2. **本地小模型**（`provider: local`）：用训练集训练 TF-IDF + 逻辑回归分类器，作为第 4 票。它单独的准确率不高（58%–73%），但出错方式与大模型不同。
3. **分歧直接转人工**（`disagreement_action: human`，现为默认）。

```bash
python scripts/prepare_econ.py      # 同时生成 data/econ_*_train.csv
python -m crosscheck evaluate data/econ_fomc_gold.csv -c configs/econ_fomc_fewshot.yaml -o output/econ_fomc_fewshot
python -m crosscheck evaluate data/econ_fomc_gold.csv -c configs/econ_fomc_hybrid.yaml  -o output/econ_fomc_hybrid
python scripts/compare_econ.py      # 输出三个版本的对比 output/econ_compare.txt
```

**全自动准确率**（不用人工，分歧样本取加权投票结果）：

| 数据集 | 基线 | 动态示例 | 动态示例 + 本地小模型 |
|---|---|---|---|
| 年报气候段落 | 83.3% | 85.3% | **86.0%** |
| FOMC 鹰鸽 | 62.0% | 65.3% | **66.0%** |
| 股吧情绪 | 68.7% | **72.0%** | 70.7% |
| 平均 | 71.3% | **74.2%** | **74.2%** |

动态示例让 9 个“模型 × 数据集”组合中的 8 个提升（最多 +4.7pp），平均 +2.9pp。单个数据集 150 条的 95% 置信区间约 ±7.7pp，所以单看一个数据集的差异不显著，但三个数据集方向一致。

**人机协作**（一致的自动采纳，其余交人工，假设人工判对）：

| 数据集 | 基线：三模型一致 | 混合：四票全一致 |
|---|---|---|
| 年报气候段落 | 自动 78.7%，准确率 91.5% → 整体 93.3% | 自动 62.7%，准确率 **95.7%** → 整体 **97.3%** |
| FOMC 鹰鸽 | 自动 76.7%，准确率 69.6% → 整体 76.7% | 自动 54.7%，准确率 **74.4%** → 整体 **86.0%** |
| 股吧情绪 | 自动 78.0%，准确率 75.2% → 整体 80.7% | 自动 53.3%，准确率 **86.2%** → 整体 **92.7%** |
| 平均 | 自动 77.8%，准确率 78.8% → 整体 83.6% | 自动 56.9%，准确率 **85.4%** → 整体 **92.0%** |

结论：

- **动态示例是稳定有效、几乎零成本的改进**，应默认开启（只要有训练集或历史标注）。
- **本地小模型的价值不在于多一票，而在于让“一致”更可信**：自动采纳部分的准确率从 78.8% 提高到 85.4%，代价是人工比例从 22% 升到 43%。可按预算选择策略：`compare_econ.py` 同时给出“四票至少三票”等折中方案（自动采纳约 90%，准确率约 79%）。
- **FOMC 的上限主要受标注尺度限制**：四票全一致仍判错的 21 条中，16 条是金标准为鹰派/鸽派、所有模型都判为中性，例如“匈牙利和波兰采用了通胀目标制”被标为鹰派。这类数据集需要更多体现标注习惯的示例，或者直接接受人工复核。

> 数据集许可：climate_sentiment 为 CC BY-NC-SA 4.0，Trillion Dollar Words 为 CC BY-NC 4.0，仅用于非商业研究。

#### 成本：费用预估、级联调用、断点续跑

**费用统计**：在 `config.yaml` 每个模型下填 `price_in` / `price_out`（元 / 百万 token，按官网价格）。运行结束后按模型显示实际请求数、缓存命中、接口返回的输入 / 输出 token、本次费用和缓存节省的费用；没填单价的模型只统计 token。

**运行前预估**：首轮提示词逐条构造并查缓存，缓存命中的不计费；后续环节取决于模型是否一致，给出“最少（全部一致）/ 预计（按分歧比例，默认 25%）/ 最多（全部分歧）”三档。

```bash
python -m crosscheck classify data/your_data.csv --estimate-only    # 只预估，不调用
python -m crosscheck classify data/your_data.csv --budget 5         # 预计费用超过 5 元就不运行
python -m crosscheck classify data/your_data.csv --resume           # 中断后续跑，跳过已完成的样本
```

**断点续跑**：每处理完一条就追加写入 `<输出目录>/<结果名>.checkpoint.jsonl`，`--resume` 时跳过已完成样本，全部完成后自动删除检查点。不加 `--resume` 时如果发现旧检查点会提示并从头开始。

**级联调用**（`pipeline.cascade`）：先只调用列出的模型（至少 2 个，少于全部投票模型）；它们全部一致就直接采纳，其余模型不再调用；不一致才调用其余模型，按正常流程投票。推荐 `cascade: [local, deepseek]`：本地小模型免费，DeepSeek 最便宜，两者出错方式不同，一致时相当可信。

在 200 条**从未用于调参的新股吧帖子**上实测（FinFE 训练集外、金标准外的样本，`configs/econ_finfe_cascade.yaml` vs `configs/econ_finfe_hybrid.yaml`）：

| 方案 | 实际费用 | 每千条 | 自动采纳 | 自动采纳准确率 | 人工兜底后 |
|---|---|---|---|---|---|
| 三个大模型全一致（原做法） | 1.65 元 | 8.3 元 | 82.5% | 81.2% | 84.5% |
| 四票全一致（混合） | 1.65 元 | 8.3 元 | 66.5% | **89.5%** | **93.0%** |
| 级联：本地 + DeepSeek 一致即采纳 | **0.66 元** | **3.3 元** | 73.5% | 87.8% | 91.0% |

- 级联只有 26.5% 的样本需要调用 Kimi 和千问，**费用降到四票方案的 40%**，人工量更少，准确率只低 2 个百分点，且明显好于原来的“三个大模型全一致”
- 在 450 条金标准上离线模拟结论一致：级联的大模型调用量为全量的 53%–60%，人工兜底后准确率 92.7% / 83.3% / 90.7%（年报气候 / FOMC / 股吧），四票全一致为 97.3% / 86.0% / 92.7%。对准确率要求最高时用四票全一致，费用敏感时用级联
- **预估准确度**：token 估算系数已按三家接口返回的实际用量校准，按实际分歧比例预估，级联 0.688 元 vs 实际 0.664 元，四票 1.702 元 vs 实际 1.651 元，误差 4% 以内

---

## 项目结构

```
├── app.py                   # Streamlit 网页平台
├── config.yaml              # 分类任务、模型、阈值配置（换任务只需改这里）
├── configs/
│   ├── tnews.yaml           # 新闻分类任务（base 继承 config.yaml 的模型配置）
│   ├── econ_climate.yaml    # 年报气候段落 风险/机遇/中性
│   ├── econ_fomc.yaml       # FOMC 货币政策 鹰派/鸽派/中性
│   ├── econ_finfe.yaml      # 股吧帖子情绪 积极/消极/中性
│   ├── econ_*_fewshot.yaml  # 上述任务 + 动态示例 + 分歧转人工
│   ├── econ_*_hybrid.yaml   # 再加本地小模型作为第 4 票（含模型单价）
│   └── econ_*_cascade.yaml  # 混合版 + 级联：本地小模型和 DeepSeek 一致即采纳
├── scripts/
│   ├── prepare_econ.py      # 下载经济金融数据集，生成金标准和训练集
│   └── compare_econ.py      # 对比基线 / 动态示例 / 混合投票三个版本
├── .env                     # API key（自行创建，不上传）
├── requirements.txt
├── data/
│   ├── gold.csv             # 客服消息金标准（60 条）
│   ├── tnews_gold.csv       # TNEWS 新闻标题金标准（150 条）
│   ├── econ_*_gold.csv      # 三个经济金融数据集金标准（各 150 条）
│   ├── econ_*_train.csv     # 对应训练集（已剔除测试集文本），用于动态示例和本地小模型
│   └── sample.csv           # 示例待分类数据
└── crosscheck/
    ├── cli.py               # 命令行入口
    ├── config.py            # 配置加载与校验
    ├── llm.py               # 模型调用层：OpenAI 兼容 / Anthropic / 本地小模型 / mock，含重试
    ├── local_model.py       # 相似样本检索（动态示例）与 TF-IDF + 逻辑回归本地分类器
    ├── prompts.py           # 首轮 / 复核 / 仲裁提示词（可附带动态示例）
    ├── classifier.py        # 结果解析、标签纠错、回复缓存、token 与费用统计
    ├── cost.py              # token 估算、运行前费用预估
    ├── pipeline.py          # 互检流水线
    ├── aggregate.py         # 加权投票、Dawid-Skene
    ├── evaluate.py          # 评估指标与权重计算
    ├── report.py            # 图表与 HTML 报告
    ├── review.py            # 人工审核：审核范围、存盘、统计、回流金标准
    └── io_utils.py          # 数据读写
```

---

## 项目规划（Roadmap）

### ✅ v0.1 基础实现（当前）

- [x] 配置驱动：类别定义、正反例、边界规则、模型、阈值全部写在 `config.yaml`
- [x] 多模型并行独立分类，结构化 JSON 输出与容错解析
- [x] 交叉复核（匿名意见 + 防盲从提示）
- [x] 加权投票 → 仲裁模型 → 人工兜底 的分层决策
- [x] 金标准评估：单模型准确率、Fleiss' Kappa、混淆矩阵、log-odds 自动权重
- [x] Dawid-Skene 无监督聚合
- [x] 回复缓存、并发控制、失败重试、mock 模式
- [x] 接入 DeepSeek / Kimi / 千问 / GLM 真实模型，`.env` 管理 key，单模型并发与 RPM 限流
- [x] 评估报告：单模型 vs 多数投票 vs 加权投票 vs 互检系统 准确率对比图、分类别对比、复核效果、混淆矩阵、HTML 报告
- [x] 配置继承（`base`），同一套模型配置复用于多个分类任务

### 🌐 v0.2 Web 平台（基础版已完成）

目标：不写命令也能用，上传文件即可得到带图表的结果。

- [x] 在页面上配置模型：填写 API key / 接口地址 / 模型名，一键测试连通性
- [x] 在页面上编辑分类任务：类别、定义、正反例、边界规则
- [x] 上传 CSV / Excel，选择文本列和标签列，实时显示进度
- [x] 结果页：各模型准确率与投票准确率对比图、分类别对比、分歧样本列表，支持下载结果
- [x] 按实测结论新增策略开关：“首轮不一致即直接仲裁 / 转人工”（`pipeline.disagreement_action`）
- [x] 人工审核页：逐条 / 批量处理“需人工审核”样本，抽检一致样本，结果回流为金标准
- [ ] 历史运行记录：查看、对比多次运行的结果
- [ ] 多次运行对比：同一份数据换模型 / 换策略后的准确率变化

### 🚧 v0.3 成本与稳定性（主体已完成）

目标：同等准确率下把成本降到 1/3 以下，并能稳定跑十万级数据。

- [x] **级联调用（Cascade）**：先调用本地小模型和最便宜的大模型，两者一致直接采纳，不一致才启动其余模型。实测费用降到 40%，准确率只低 2 个百分点
- [x] **断点续跑**：按样本增量写检查点，`--resume` 跳过已完成样本
- [x] **分模型限流**：每个模型独立的并发数和 RPM 限制（TPM 限制暂未做）
- [x] **Token 与费用统计**：按模型统计接口返回的输入 / 输出 token、费用和缓存节省
- [x] **运行前费用预估与预算上限**：命令行 `--estimate-only` / `--budget`，网页“预估调用次数和费用”按钮
- [x] **输出容错**：模型在理由里写未转义的双引号导致 JSON 不合法时，逐字段提取标签
- [ ] **置信度级联**：对支持 logprobs 的模型，按标签概率决定是否升级，而不只看是否一致
- [ ] **选项顺序随机化**：消除模型对靠前选项的位置偏好
- [ ] **Excel 输入输出**、单元测试与 CI

### 🔬 v0.4 更强的聚合与校准

目标：让“置信度”真正可信，让投票更聪明。

- [ ] **按类别的权重**：用混淆矩阵代替单一权重。例如模型 A 判“投诉”很准但判“建议”很差，则只在“投诉”上给它高权重
- [ ] **置信度校准**：模型自报的置信度普遍偏高，用金标准做 Temperature Scaling / Isotonic 回归校准，让 0.8 真正意味着 80% 正确
- [ ] **Self-Consistency**：同一模型多次采样（temperature > 0）投票，用答案稳定性作为置信度，比自报置信度更可靠
- [ ] **Logprobs 置信度**：对支持 logprobs 的模型，直接读取标签 token 的概率
- [ ] **更多聚合算法**：接入 [crowd-kit](https://github.com/Toloka/crowd-kit) 的 GLAD、MACE、Wawa 等，自动选择最优聚合方式
- [ ] **动态阈值**：根据目标准确率（例如 98%）在金标准上自动搜索 `accept_threshold` 与 `arbiter_threshold`

### 🗣️ v0.5 高级协作模式

目标：在难例上逼近人类专家水平。

- [ ] **多轮辩论**：分歧样本进行 N 轮辩论直到收敛，参考 [llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate)
- [ ] **角色化评审**：引入“魔鬼代言人”角色，专门寻找当前多数意见的漏洞，对抗从众效应
- [ ] **证据引用**：要求模型在理由中逐字引用原文作为证据，没有证据支撑的判断降权
- [ ] **对照式复核**：复核者不看结论只看理由，或只看结论不看理由，减少锚定效应
- [ ] **评审团模式（Jury）**：用多个便宜小模型组成评审团代替单个大模型仲裁，参考论文 *Replacing Judges with Juries (PoLL)*
- [ ] **多标签 / 层级分类**：支持一条文本多个标签、一级类 → 二级类的层级结构

### 🔁 v0.6 闭环自进化

目标：越用越准，人工工作量持续下降。

- [ ] **人工结果回流**：人工审核后的样本自动进入样本库
- [x] **动态 Few-shot**：分类时检索最相似的已标注样本作为示例放进提示词（当前为字符级 TF-IDF，`fewshot` 配置）
- [ ] 动态 Few-shot 升级为向量检索（embedding），并让人工审核结果自动进入检索库
- [ ] **分歧模式分析**：定期让 LLM 汇总分歧样本，自动归纳“哪条规则写得模糊”，并给出分类标准修改建议
- [ ] **提示词自动优化**：用 [DSPy](https://github.com/stanfordnlp/dspy) 等工具在金标准上自动搜索最优提示词和示例组合
- [ ] **主动学习**：优先把“对模型提升最大”的样本推给人工标注，而不是随机抽样
- [ ] **标签噪声检测**：用 [cleanlab](https://github.com/cleanlab/cleanlab) 找出金标准和历史结果中可能标错的样本

### 🏭 v0.7 蒸馏与工程化

目标：从“实验工具”变成“生产系统”。

- [x] **本地小模型投票**：用训练集训练 TF-IDF + 逻辑回归分类器作为异构投票者（`provider: local`）
- [ ] **小模型蒸馏**：用互检得到的高置信数据训练 BERT / 小尺寸开源模型，日常流量由小模型处理，互检系统只负责难例和持续产出训练数据
- [ ] **人工审核界面**：基于 Streamlit 的轻量审核页面，或对接 [Label Studio](https://github.com/HumanSignal/label-studio) / [Argilla](https://github.com/argilla-io/argilla)
- [ ] **API 服务**：FastAPI 封装，支持同步单条和异步批量
- [ ] **质量监控**：跟踪一致率、人工率、各模型准确率随时间的漂移，模型升级或数据分布变化时报警
- [ ] **本地模型支持**：接入 Ollama / vLLM，敏感数据不出内网

### 📄 v0.8 长文档：上传企业年报自动分类

目标：上传一份（或一批）企业年报，自动切分、逐段分类，汇总成公司-年度指标，并能像信息抽取审查页那样对照原文审核。

- [ ] **文档解析**：支持 PDF / Word / HTML 年报（巨潮资讯下载的 PDF 为主），提取正文和页码；去掉目录、页眉页脚、重复声明；表格单独处理；扫描件可选 OCR
- [ ] **章节识别与切分**：识别“管理层讨论与分析”“风险因素”“环境和社会责任”等章节，按段落 / 句子切分，保留所在章节和页码，只对选定章节分类以节省费用
- [ ] **逐段互检分类**：复用现有流水线；支持一段多个标签（如同时属于“气候风险”和“前瞻性表述”）；可接入年报训练数据做动态示例
- [ ] **成本控制**：运行前按段落数和 token 估算费用；先用本地小模型或关键词过滤掉明显无关的段落，只把候选段落交给大模型
- [ ] **文档级汇总**：输出每份年报的指标，例如各类段落数量与占比、净语调 =（积极 − 消极）/ 总段落数，批量导出公司 × 年度面板数据，供计量分析直接使用
- [ ] **信息抽取模式**：除了分类，还能按字段抽取结构化信息（如风险事项、减排目标、研发投入表述），每条结果附原文依据和页码
- [ ] **原文对照审核页**：左侧显示年报原文并高亮被分类 / 抽取的段落，右侧列出结果；有标准答案时标出“对上 / 漏掉 / 多提取”；点击条目定位到原文；审核结果回流训练数据
- [ ] **批量处理**：一次上传多份年报（或给出股票代码 + 年份自动下载），断点续跑，按公司和年份管理结果

---

## 注意事项

1. **模型要异构**：同一家族的模型（例如 GPT-4o 和 GPT-4o-mini）错误高度相关，互检效果会打折扣，建议选不同厂商。
2. **分类标准比模型更重要**：分歧集中在某两个类别之间时，优先修改边界规则，而不是换模型。
3. **“一致”不等于“正确”**：所有模型可能因为同一个误解而一致答错，建议对“首轮一致通过”也做少量随机抽检。
4. **不要把 API key 写进配置文件或提交到仓库**，统一使用环境变量。
5. 部分新模型不接受 `temperature` 或 `max_tokens` 参数，遇到 400 错误时请调整对应模型的配置。

---

## 参考项目与论文

| 项目 | 说明 |
|---|---|
| [Toloka/crowd-kit](https://github.com/Toloka/crowd-kit) | 众包标注聚合算法库（Dawid-Skene、GLAD、MACE 等） |
| [snorkel-team/snorkel](https://github.com/snorkel-team/snorkel) | 弱监督框架，Label Model 聚合多个噪声标注源 |
| [cleanlab/cleanlab](https://github.com/cleanlab/cleanlab) | 置信学习，检测标签错误 |
| [refuel-ai/autolabel](https://github.com/refuel-ai/autolabel) | 用 LLM 做数据标注，支持置信度估计 |
| [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate) | 多智能体辩论提升事实性与推理 |
| [togethercomputer/MoA](https://github.com/togethercomputer/MoA) | Mixture-of-Agents 多模型分层协作 |
| [yuchenlin/LLM-Blender](https://github.com/yuchenlin/LLM-Blender) | 多 LLM 集成（排序 + 融合） |
| [thunlp/ChatEval](https://github.com/thunlp/ChatEval) | 多智能体评审团 |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 提示词自动优化 |

- Dawid & Skene, *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*, 1979
- Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, 2023
- Verga et al., *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models*, 2024
