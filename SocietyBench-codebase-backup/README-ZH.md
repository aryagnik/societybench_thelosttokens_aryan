<div align="center">

# 🔮 SocietyBench

反事实社会世界的演化预测
</br>
Forecasting Counterfactual Social-World Evolution

[![License](https://img.shields.io/badge/License-MIT-2a78d6?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-1baf7a?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SocietyBench-eda100?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)
[![Events](https://img.shields.io/badge/Events-5-eb6834?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)
[![Questions](https://img.shields.io/badge/Questions-25.4k%20%C3%97%202-e87ba4?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)

[English](./README.md) | [中文文档](./README-ZH.md)

</div>

## ⚡ 项目概述

**SocietyBench** 衡量的是"完成任务类"评测碰不到的一种能力：模型能不能预测一件真实社会事件**接下来怎么发展**。

在真实事件上做这件事有个绕不开的麻烦——强模型可能根本不是在预测，而是在**回忆**：它在预训练数据里见过这件事。SocietyBench 把这条捷径堵死。在任何模型看到时间线之前，每一个实体名都会被替换成占位符，每一个日期都会整体平移一个该事件专属的常数。因果结构和时间结构完好保留，能让模型对上记忆的表面标签则全部消失。剩下的就是一个**反事实社会世界**——一条只能被预测、无法被回忆的演化轨迹。

> **你提供**：一句话的事件主题。</br>
> **流水线返回**：一份匿名化的"事实 + 舆论"时间线、每个截止日期上一套经过审核的题库，以及两条正交评分轴上的分数。

本仓库是完整流水线：采集原始素材、构建时间线、匿名化、生成题目、给答案打分。

## 🎯 一图看懂

模型被放在一个**截止日期**上。它能看到这个日期之前的一切，看不到之后的任何东西，然后回答两类关于"接下来会发生什么"的问题：

| 评分轴 | 问题形式 | 指标 | 平凡基准 |
|--------|----------|------|----------|
| **概率校准** | 事件 *E* 会在未来 *W* 天内发生吗？ | 加权 MAE → 0–100 | 所有题都答 50% 恰好得 **50** 分 |
| **时间准确度** | 事件 *e* 具体在哪一天发生？ | 分段归一化的天数误差 → 0–100 | 每次都猜区间中点约得 **50** 分 |

| | 规模 |
|---|---|
| **事件数** | 5 —— 公共舆论、地缘政治、科技监管、金融市场、贸易政策 |
| **预测点** | 每个事件 25 个，共 **125** 个 |
| **校准题** | 每个语言版本 **25,364** 题 |
| **时间事件** | 每个语言版本 **3,112** 个 |
| **语言版本** | 中文、英文，逐条一一对应 |
| **已评测系统** | 6 个前沿 LLM · 3 个智能体框架 · 2 个无模型基线 |

最强的模型只拿到 **75.0 / 100**——大约只收回了平凡基准之上一半的空间。这个榜远没有被刷满。

## 🔄 工作流程

1. **采集** —— 从一句话主题出发，抓取五个平台的网络新闻与社交媒体帖子
2. **蒸馏** —— 一条智能体链把原始条目压缩成按日期索引的时间线，并把**事实**事件与**舆论**层分开保存
3. **匿名** —— 三个阶段：实体替换与日期平移、对抗式反向识别审计、语义一致性修复
4. **出题** —— 每个截止日期都生成一套经过审核的校准题与时间题
5. **评分** —— 两条独立的百分制评分轴，先按事件计算，再跨事件平均

## 🚀 快速开始

### 前置要求

| 工具 | 版本要求 | 说明 | 安装检查 |
|------|---------|------|---------|
| **Python** | 3.10+ | 全流程运行环境 | `python3 --version` |
| **LLM 接口** | 任意 OpenAI 兼容 | 被测模型（DMXAPI、OpenRouter 等） | — |
| **Apify token** | 可选 | 只有自己抓取网络新闻时才需要 | — |

#### 1. 安装

```bash
git clone https://github.com/co-minder/SocietyBench-codebase
cd SocietyBench-codebase

pip install -r requirements.txt          # 评测
pip install -r requirements-crawl.txt    # 只有需要自己抓数据时才装
pip install -r requirements-agents.txt   # 只有需要跑智能体基线时才装
```

#### 2. 配置

```bash
cp eval/config.example.env eval/.env
```

**必需的环境变量：**

```env
# 任意 OpenAI 兼容接口均可（DMXAPI、OpenRouter 等）
DMXAPI_KEY=sk-...
DMXAPI_BASE_URL=https://www.dmxapi.com/v1
```

在启动任何长跑任务之前先验一下：

```bash
python3 eval/health_check.py     # 预期输出：[health] OK — model=...
```

#### 3. 复现论文里的数字

不需要抓数据——这条命令会从数据仓库拉取一个事件，并端到端跑完两条评分轴：

```bash
python3 main.py --reproduce event3_tiktok /path/to/workspace
```

可选事件：`event1_library` · `event2_trump_tariff` · `event3_tiktok` · `event4_us_iran` ·
`event5_smci`。结果落在 `<workspace>/results/run_<timestamp>/`。

数据已经在本地了？直接调评测器：

```bash
python3 eval/run_pipeline_parallel.py \
    --workspace /path/to/event3_tiktok/zh \
    --event-name event3_tiktok \
    --models "<model-id>"
```

> **英文版本注意**：脚本是靠"工作目录名是否为 `英文`"来切换英文 prompt 的。要评测 `<event>/en`，
> 请先做一个软链接 `ln -s en 英文`，再把这个路径传进去。

#### 4. 用你自己的事件造一个榜

```bash
# 告诉它哪些真实名字映射到哪些占位符
cat > reps.json <<'JSON'
{"特斯拉": "公司A", "马斯克": "人物A", "奥斯汀": "城市A"}
JSON

python3 main.py "2025年特斯拉德州工厂罢工" /path/to/workspace \
    --replacements-json reps.json
```

加 `--start-phase N` 可以从中途续跑：`0` 从抓取开始，`1` 从单源处理开始，`2` 从合并开始，
`3` 只跑评测。

## 🏗️ 项目结构

| 路径 | 内容 |
|------|------|
| `main.py` | 入口——复现某个事件，或者造一个新的 |
| `eval/` | 每个流水线阶段一个脚本；不含任何事件专属内容 |
| `eval/predict_step0–2*` | 匿名化、预测点筛选、题库生成 |
| `eval/predict_step3B*` | 概率校准轴 |
| `eval/predict_step3F*` | 时间准确度轴 |
| `eval/predict_step4_scorecard*` | 跨事件汇总 |
| `eval/agents/` | 三个智能体基线——LangGraph、AutoGen、MiroFish |
| `eval/baseline/` | 两个无模型启发式——事件基础发生率、7 日动量 |
| `eval/pipeline_config.json` | 运行参数：模型、阈值、评分常数 |
| `eval/config.example.env` | 凭据模板——复制成 `.env` 再填 |
| `docs/` | 方法、评分、架构、用法、案例、FAQ |

> 两个配置文件的分工不要混：`pipeline_config.json` 是通用行为，该进版本控制；
> `.env` 装的是你的密钥，永远不该进。

## 📚 文档

| 文档 | 回答什么问题 |
|------|-------------|
| [项目结构](docs/project_structure.md) | 完整文件索引——每个文件是干什么的 |
| [方法](docs/methodology.md) | 框架、匿名化、两条评分轴 |
| [流水线架构](docs/pipeline_architecture.md) | 数据流、工作目录布局、配置项 |
| [评分](docs/scoring.md) | 校准与时间两个公式的具体定义 |
| [案例研究](docs/case_studies.md) | 五个事件，以及怎么跑一个新事件 |
| [用法](docs/usage.md) | 更细的使用说明 |
| [常见问题](docs/faq.md) | 换模型、排错、常见报错 |

## 💾 数据

匿名时间线、题库与标准答案：
**[🤗 Social-AI-2026/SocietyBench](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)**

实体替换表**刻意不发布**。放出来就等于把每个事件都反匿名化，这个榜也就没有意义了。

## 🤝 参与贡献

欢迎提交新的模型适配、智能体基线，或者用本流水线构造的新事件。请先读
[CONTRIBUTING.md](CONTRIBUTING.md)，其中有两条硬规矩：**不要提交任何真实实体材料**，
**不要提交任何密钥**。

如果你发现的是匿名化泄漏（残留真名、可被搜索识别的细节等），请不要公开提 issue——
按 [SECURITY.md](SECURITY.md) 走私密通道，公开指认底层真实事件会污染整个榜单。

## 📄 引用

```bibtex
@misc{societybench2026,
  title  = {SocietyBench: Forecasting Counterfactual Social-World Evolution},
  author = {Wang, Zhenran and Bian, Zhonghan and Li, Jinsong and Qi, Zhangyang},
  year   = {2026},
  note   = {\url{https://github.com/co-minder/SocietyBench-codebase}}
}
```

机器可读的元数据在 [`CITATION.cff`](CITATION.cff)，版本变更记录在 [`CHANGELOG.md`](CHANGELOG.md)。

## 🙏 致谢

MiroFish 智能体基线基于 **[MiroFish](https://github.com/666ghj/MiroFish)** 构建，
其仿真引擎由 CAMEL-AI 团队的 **[OASIS](https://github.com/camel-ai/oasis)** 驱动。
另外两个智能体基线分别使用 **[LangGraph](https://github.com/langchain-ai/langgraph)** 和
**[AutoGen](https://github.com/microsoft/autogen)**。衷心感谢以上团队的开源贡献！

## ⚖️ 许可

代码采用 [MIT 许可证](LICENSE)。评测数据单独发布，采用 CC BY 4.0。
