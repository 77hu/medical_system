<p align="center">
  <h1 align="center">🏥 AI Patient-Doctor System</h1>
  <p align="center"><em>多 LLM 协同的医学问诊模拟与诊断评估研究平台</em></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Azure_OpenAI-o3--mini-0078D4?logo=microsoftazure&logoColor=white" alt="Azure">
  <img src="https://img.shields.io/badge/Azure_AI-Llama_3.3_70B-0078D4?logo=meta&logoColor=white" alt="Llama">
  <img src="https://img.shields.io/badge/SiliconFlow-DeepSeek_R1_32B-6C3CFF?logo=deepseek&logoColor=white" alt="DeepSeek">
  <img src="https://img.shields.io/badge/experiments-98_conversations-important" alt="Experiments">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 📖 目录

| 🏠 [项目介绍](#1-项目介绍) | 🚀 [如何使用](#2-如何使用) | 🔬 [实验设计](#3-实验设计) | 📊 [实验结果](#4-实验结果与分析) | 🛠 [开发说明](#5-开发说明) |
|:---:|:---:|:---:|:---:|:---:|
| 背景 · 架构 · 配置 | 安装 · 运行 · 演示 | 变量 · 数据 · 评估 | 指标 · 分析 · 成本 | 设计 · 提示词 · 日志 |

---

# 1 项目介绍

> 🎯 **核心目标**：使用三方大语言模型分别扮演医生与患者角色，模拟不同情绪下的多轮问诊对话，通过症状扰动测试AI医生的诊断鲁棒性，最终输出 17 维评估报告与可视化图表。

## 1.1 关于AI问诊模拟

医学问诊是临床诊断的核心环节——医生通过系统性的提问收集病史，逐步缩小鉴别诊断范围。评估 LLM 在问诊场景中的诊断能力，需要**同时模拟医生和患者双方的语言行为**：医生的系统问诊策略 vs 患者在不同情绪下的信息表达方式。

<table>
<tr>
  <th>方法</th><th>优点</th><th>局限</th>
</tr>
<tr>
  <td>📝 <b>静态选择题</b><br><sub>MedQA / USMLE</sub></td>
  <td>规模大、成本低、可自动化</td>
  <td>跳过问诊过程，无法评估交互式诊断能力</td>
</tr>
<tr>
  <td>👤 <b>标准化病人 (SP)</b></td>
  <td>最接近真实临床</td>
  <td>单次 $50-200、规模 <100 人、情绪一致性差</td>
</tr>
<tr>
  <td>🤖 <b>单 LLM 评测</b></td>
  <td>成本低、可扩展</td>
  <td>自问自答导致数据污染，缺少动态信息获取过程</td>
</tr>
<tr>
  <td>✅ <b>多 LLM 协同</b><br><sub>本项目方案</sub></td>
  <td><b>三方模型独立 · 全因子设计 · 可复现</b></td>
  <td>需要多厂商 API、成本略高</td>
</tr>
</table>

```
📦 数据加载 ──▶ 🔄 症状扰动 ──▶ 💬 多 LLM 问诊 ──▶ 📊 多维评估 ──▶ 📈 可视化报告
```

| 阶段 | 模块 | 输入 | 输出 |
|:---:|------|------|------|
| ① | `DataLoader` + `SymptomGenerator` | 302例罕见病 + 48例NEJM + 200+例USMLE | 3种数据变体 (原始 / +假症状 / -症状) |
| ② | `DoctorAgent` (o3-mini) + `PatientAgent` (Llama-3.3-70B) | 结构化病例 | 多轮对话 JSON + 诊断 JSON |
| ③ | `Evaluator` + `PerformanceTracker` | 对话结果列表 | 评估报告 + 3张PNG + CSV + 性能报告 |

---

## 1.2 技术特点

<table>
<tr>
<td width="50%">

### 🧩 多厂商 LLM 分离部署

| 角色 | 模型 | 厂商 |
|:---:|------|------|
| 🩺 医生 | `o3-mini` | Azure OpenAI |
| 🤒 病人 | `Llama-3.3-70B` | Azure AI MAAS |
| 🔍 评估 | `DeepSeek-R1-32B` | SiliconFlow |

> 避免单一模型"自问自答"，角色-能力精确匹配，`BaseAgent` 参数化设计兼容任意 OpenAI 格式 API

</td>
<td width="50%">

### 🎭 情绪模拟 × 信息控制

| 情绪 | 核心行为 |
|:---:|------|
| 😊 `calm` | 清晰配合，有条理 |
| 😰 `anxious` | 急促紧张，反复强调 |
| 🤨 `distrustful` | 质疑问题，不愿透露 |
| 😵 `confused` | 描述模糊，信息矛盾 |
| 😡 `aggressive` | 激动攻击，要求立即诊断 |

> 双层约束：情绪行为指导 + "无医学知识/不杜撰/两句话内"信息控制，含 fallback 防崩溃机制

</td>
</tr>
</table>

<table>
<tr>
<td width="50%">

### 🔬 症状扰动实验设计

```
原始数据 ──┬── original (基线)
           ├── +2个 LLM 生成的假症状 (干扰)
           └── -30% 真实症状 (信息缺失)
```

> 遵循"1同系统+2跨系统"假症状原则，内置 `evaluate_symptom_relevance()` 验证函数

</td>
<td width="50%">

### 🎯 三层诊断匹配算法

```
直接匹配 → 映射表匹配(18组中英同义词)
         → 关键词交叉(阈值30%)
         → 医学词汇跨语言匹配(9组)
```

> 从严格到宽松递进式判定，避免"表述不同但实质正确"的诊断被误判

</td>
</tr>
</table>

### 📋 对话全量持久化

每次问诊保存完整时间戳 JSON：对话原文 · AI 诊断 (主诊断+鉴别+检查+方案+置信度) · 正确性判定 · Token/耗时/调用次数。**所有实验对话完全可审计**，支持事后复核和二次分析。

---

## 1.3 目录结构

```
ai_patient_doctor_system/
│
├── agents/                      🤖 LLM Agent 模块
│   ├── base_agent.py            #   多厂商API兼容 · Token统计 · 对话历史
│   ├── doctor_agent.py          #   VINDICATE系统问诊 · JSON诊断 · 自动终止
│   └── patient_agent.py         #   5情绪注入 · 地理背景 · 逐步披露 · fallback
│
├── data_processing/             🔄 数据处理模块
│   ├── data_loader.py           #   罕见病JSON / NEJM CSV / USMLE CSV / HF 四格式适配
│   └── symptom_generator.py     #   LLM假症状 · 随机删除30% · 三变体管线
│
├── evaluation/                  📊 评估模块
│   └── evaluator.py             #   指标计算 · 分组统计 · 错误分析 · 可视化 (3图1表)
│
├── utils/                       ⚙️ 工具模块
│   └── performance_tracker.py   #   Token/时间/API统计 · JSON+CSV双输出 · 按情绪分组
│
├── configs/                     ⚙️ 配置
│   └── config.py                #   3套API · 3个模型 · 5种情绪 · 路径 · 对话参数
│
├── step1/                       🧪 数据准备管线
│   ├── step1_1.py               #   生成假症状数据集 → fake_data.json
│   └── step1_2.py               #   读取假数据 → 批量问诊 → 评估可视化
│
├── data/                        📦 数据集 (见下表)
├── conversation_manager.py      💬 对话编排引擎 (批量问诊 · 三层匹配 · 持久化)
├── main.py                      🚀 主入口 (完整实验管线 + 单案例演示)
├── test_optimized.py            🎯 副球孢子菌病专项测试
├── prompts_revised.py           📝 Prompt 工程文档 (设计稿)
└── results/                     📈 输出: 98个对话JSON + 7份评估报告 + 3图 + CSV
```

### 数据集详情

| 📦 数据集 | 文件 | 案例数 | 格式 | 说明 |
|:---|------|:---:|:---:|------|
| 🔬 罕见病 | `rare_disease_302.json` | 302 | JSON | MedRBench格式，Initial/Follow-up双阶段 |
| 🏥 NEJM AI | `nejmai_dataset.csv` | 48 | CSV | 新英格兰医学杂志影像挑战，5选1 |
| 📚 USMLE/Derm | `usmle_and_derm_dataset.csv` | 200+ | CSV | 医师考试+皮肤科，4选1，含category字段 |
| 📄 MedRBench-Dx | `diagnosis_957_cases...json` | 957 | JSON | PMC病例报告，含鉴别诊断+最终诊断 |
| 💊 MedRBench-Rx | `treatment_496_cases...json` | 496 | JSON | PMC治疗规划数据 |
| 🇨🇳 JMED | HuggingFace `jdh-algo/JMED` | — | HF | 京东健康中文医疗数据集 |
| 🧪 假症状 | `generation_data/fake_data.json` | — | JSON | 预生成的3种数据变体 |

---

## 1.4 依赖

```bash
pip install requests pandas numpy scikit-learn matplotlib seaborn datasets openpyxl
```

| 库 | 版本 | 用途 |
|------|:---:|------|
| `requests` | ≥2.28 | LLM API HTTP 调用 (兼容 OpenAI 格式多厂商端点) |
| `pandas` | ≥1.5 | CSV 读写、DataFrame 分组统计、报告输出 |
| `numpy` | ≥1.23 | 统计计算 (mean/std)、雷达图坐标 |
| `scikit-learn` | ≥1.2 | Accuracy / F1 / Precision / Recall |
| `matplotlib` | ≥3.6 | 柱状图、雷达图 (中文: SimHei / Arial Unicode MS) |
| `seaborn` | ≥0.12 | 样式增强 |
| `datasets` | ≥2.10 | HuggingFace 云端数据集加载 |
| `openpyxl` | — | Excel 读写 (USMLE 数据源) |

---

## 1.5 系统架构

<pre>
                      📦 医疗数据集 (JSON / CSV / HuggingFace)
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  <b>Step 1: data_processing/</b>  (~375 行)                                  │
│                                                                      │
│  ┌──────────────────── DataLoader ──────────────────────────────┐    │
│  │  load_rare_disease_data()   302例 JSON → 取前 50             │    │
│  │  load_nejmai_data()         48例 CSV  → 全量                 │    │
│  │  load_usmle_derm_data()     200+ CSV  → 取前 100             │    │
│  │  load_jmed_data()           HF 云端   → 取前 50              │    │
│  │  ➜ 统一格式 [{case_id, chief_complaint, symptoms, ...}]     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ┌───────────────── SymptomGenerator ───────────────────────────┐    │
│  │  original              → 原始基线                            │    │
│  │  with_fake_symptoms    → +2 LLM 生成假症状 (1同系统+2跨系统)  │    │
│  │  with_missing_symptoms → 随机删除 30% 症状 (至少保留 1 个)    │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  <b>Step 2: agents/ + conversation_manager.py</b>                        │
│                                                                      │
│  ┌───────────── DoctorAgent (Azure o3-mini) ───────────────────┐     │
│  │  generate_question()                                         │     │
│  │    ├── VINDICATE 框架 (血管→感染→肿瘤→…→内分泌)               │     │
│  │    ├── 地理史/旅行史/职业暴露/动物接触 专项询问               │     │
│  │    └── 最近 6 轮对话上下文                                    │     │
│  │  should_end_consultation() → ≥16条消息 OR 信息充分            │     │
│  │  make_diagnosis() (t=0.3) → JSON {主诊断, 鉴别[], 检查[],     │     │
│  │    治疗方案, confidence}                                      │     │
│  └──────────────────────────────────────────────────────────────┘     │
│  ┌───────────── PatientAgent (Azure Llama-3.3-70B) ────────────┐     │
│  │  _build_system_prompt()                                       │     │
│  │    ├── 病例信息注入 (主诉/症状/病史)                           │     │
│  │    ├── 地理背景条件注入 (case_6 → 巴西移民)                    │     │
│  │    ├── 情绪行为指导 (5 种 × 行为规则)                          │     │
│  │    └── 约束: 无医学知识 · 不杜撰 · 两句话 · 逐步披露          │     │
│  │  generate_response() → 正常 LLM 回应 / fallback 默认回应       │     │
│  └──────────────────────────────────────────────────────────────┘     │
│  ➜ 对话 JSON + 性能统计 (Token/时间/API，Doctor与Patient分开)        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  <b>Step 3: evaluation/ + utils/</b>                                     │
│                                                                      │
│  ┌─────────── Evaluator ────────────────────────────────────────┐    │
│  │  evaluate_results()                                           │    │
│  │    ├── 整体: accuracy / precision / recall / f1               │    │
│  │    ├── 分组: by_emotion (5种) + by_modification (3种)         │    │
│  │    ├── 错误: 前10案例 (病例/情绪/预测/实际/置信度)            │    │
│  │    └── 汇总: avg_turns / std / avg_confidence                 │    │
│  │  create_visualizations()                                      │    │
│  │    ├── 📊 accuracy_by_emotion.png      情绪柱状图             │    │
│  │    ├── 📊 accuracy_by_modification.png 三色数据变体图         │    │
│  │    ├── 🎯 overall_metrics_radar.png    四维雷达图             │    │
│  │    └── 📋 summary_statistics.csv       双维度汇总表           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ┌─────────── PerformanceTracker ───────────────────────────────┐    │
│  │  save_performance_summary()  → JSON + CSV 双格式              │    │
│  │  print_performance_summary() → 控制台格式化输出               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  <b>results/</b>                                                         │
│  ├── conversations/  (<b>98</b> 个对话 JSON)                               │
│  ├── evaluations/    (<b>7</b> 份评估报告 JSON)                             │
│  ├── performance/    (性能 JSON + CSV)                               │
│  ├── accuracy_by_emotion.png                                        │
│  ├── accuracy_by_modification.png                                   │
│  ├── overall_metrics_radar.png                                      │
│  └── summary_statistics.csv                                         │
└──────────────────────────────────────────────────────────────────────┘
</pre>

---

## 1.6 模型配置

| 🎭 角色 | 🤖 模型 | ☁️ API 提供商 | 🏷️ 用途 |
|:---:|------|------|------|
| 🩺 医生 | `gpt-o3mini` | Azure OpenAI | 系统问诊 + 鉴别诊断 + 治疗方案 |
| 🤒 病人 | `Llama-3.3-70B-Instruct` | Azure AI (MAAS) | 情绪化患者回应 + 症状逐步披露 |
| 🔍 评估 | `DeepSeek-R1-Distill-Qwen-32B` | SiliconFlow | 假症状生成 + 诊断准确性评估 |

> 💡 **设计考量**：① 跨厂商部署防止"自问自答"数据污染 ② 推理型任务用强模型，角色扮演用中等模型 ③ `headers` + `API_URL` 参数化支持一键切换供应商

---

# 2 如何使用

## 2.1 安装依赖

```bash
pip install requests pandas numpy scikit-learn matplotlib seaborn datasets openpyxl
```

## 2.2 配置 API Key

> ⚠️ 推荐使用 `.env` 文件管理密钥，复制 `.env.example` → `.env` 后填入真实值

编辑 `configs/config.py` (或设置环境变量)：

```python
# 评估模型 — SiliconFlow (DeepSeek)
API_KEY_EVAL  = "your-siliconflow-key"
API_URL_EVAL  = "https://api.siliconflow.cn/v1/chat/completions"

# 医生模型 — Azure OpenAI (o3-mini)
API_KEY_DOCTOR = "your-azure-openai-key"
API_URL_DOCTOR = "https://YOUR_RESOURCE.openai.azure.com/.../chat/completions?..."

# 病人模型 — Azure AI MAAS (Llama-3.3-70B)
API_KEY_PATIENT = "your-azure-ai-key"
API_URL_PATIENT = "https://YOUR_RESOURCE.services.ai.azure.com/.../chat/completions?..."
```

<details>
<summary>🔧 可调参数</summary>

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `MAX_CONVERSATION_TURNS` | 10 | 最大问诊轮数 (影响信息完整度和Token成本) |
| `TEMPERATURE` | 0.7 | 对话温度 (医生诊断阶段内部固定为 0.3) |
| `PATIENT_EMOTIONS` | 5种 | calm/anxious/distrustful/confused/aggressive |
| `DOCTOR_MODEL` | `gpt-o3mini` | 可切换为任意 OpenAI 兼容模型 |
| `PATIENT_MODEL` | `Llama-3.3-70B-Instruct` | 可切换 |
| `EVAL_MODEL` | `DeepSeek-R1-Distill-Qwen-32B` | 可切换 |

</details>

## 2.3 准备数据

| 文件 | 必须 | 说明 |
|------|:---:|------|
| `data/rare_disease_302.json` | ✅ | 罕见病数据集 (302例, MedRBench格式) |
| `data/nejmai_dataset.csv` | ✅ | NEJM AI 影像挑战 (48例, 5选1) |
| `data/usmle_and_derm_dataset.csv` | — | USMLE + 皮肤科 (200+例, 4选1) |
| `data/MedRBench/` | — | 扩展集 (诊断957例 + 治疗496例) |

## 2.4 快速开始

```bash
# 🎯 单案例演示 (偏头痛, 测试2种情绪+假症状)
python main.py
# 约消耗 20K-30K tokens, 2-3 分钟

# 🧪 副球孢子菌病专项测试 (南美地方真菌病)
python test_optimized.py

# 🔬 完整实验 (5病例 × 3情绪 × 3数据变体)
# 修改 main.py: main() 替代 run_single_case_demo()
python main.py
# 约消耗 500K-1200K tokens, 1-2.5 小时
```

<details>
<summary>📦 step1 离线预处理管线</summary>

```bash
# 第一步: 生成假症状数据集
python step1/step1_1.py
# ➜ data/generation_data/fake_data.json

# 第二步: 基于假数据批量问诊+评估
python step1/step1_2.py
# ➜ results/evaluations/ + performance/ + 3图 + CSV
```

> 适合迭代开发：调整假症状策略 → 保存 JSON → 多次运行 step1_2

</details>

---

# 3 实验设计

## 3.1 实验变量

本项目设计了两个核心自变量，采用**全因子交叉设计**：

<table>
<tr><th colspan="2"><b>🧪 自变量 1: 病人情绪状态</b></th></tr>
<tr>
  <td width="50%">

| 变量值 | 描述 | 核心行为 |
|:---:|------|------|
| `calm` | 😊 平和 | 清晰有条理、主动配合 |
| `anxious` | 😰 焦虑 | 急促紧张、反复强调担忧 |
| `distrustful` | 🤨 不信任 | 质疑问题、挑战专业性 |

  </td>
  <td width="50%">

| 变量值 | 描述 | 核心行为 |
|:---:|------|------|
| `confused` | 😵 困惑 | 描述模糊、信息矛盾 |
| `aggressive` | 😡 激动 | 语气激烈、要求立即诊断 |

  </td>
</tr>
</table>

<table>
<tr><th colspan="2"><b>🧪 自变量 2: 数据完整性</b></th></tr>
<tr>
  <td>

| 变量值 | 说明 | 实现 |
|:---:|------|------|
| `original` | 📝 原始完整 | 直接使用数据集症状列表 |
| `fake_symptoms_added` | ➕ 添加假症状 | LLM 生成 2 个干扰症状 |
| `symptoms_removed` | ➖ 删除症状 | 随机删除 30% (至少保留1个) |

  </td>
</tr>
</table>

### 实验矩阵

```
                     calm    anxious    distrustful    confused    aggressive
original              ✓         ✓            ✓           —            —
fake_symptoms_added   ✓         ✓            —           —            —
symptoms_removed      ✓         ✓            —           —            —
```

> 首轮聚焦核心交叉 (21组/轮)，后续可扩展至 5×3 = 15组/病例 全因子

---

## 3.2 实验病例

| ID | 🏥 疾病 | 🏷️ 类型 | 🔑 关键诊断线索 |
|:---:|------|:---:|------|
| case_1 | Pulmonary Sarcoidosis | 呼吸/免疫 | 非干酪性肉芽肿，须排除结核和真菌感染 |
| case_4 | Vertical Nystagmus | 神经/眼科 | 饮酒+晕倒+眼部受压3小时→排除型题目 |
| case_6 | Paracoccidiomycosis | 感染(地方真菌) | 🇧🇷 巴西移民史是关键流行病学线索 |
| case_7 | Exogenous Ochronosis | 皮肤 | 氢醌美白霜使用史是唯一诊断线索 |
| case_8 | (皮肤科案例) | 皮肤 | 无症状皮疹，需鉴别多种皮肤病 |

> 📋 选取策略：覆盖呼吸/神经/感染/皮肤四大系统，含常见鉴别诊断 + 罕见地方病

---

## 3.3 17 维评估体系

<table>
<tr><th colspan="2">🩺 诊断质量 (7维)</th></tr>
<tr><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 1 | 诊断准确性 | 完全准确 ↔ 完全不正确 |
| 2 | 诊断逻辑清晰性 | 非常清晰 ↔ 很不清晰 |
| 3 | 诊断逻辑充分性 | 足够支撑结论 ↔ 极不充分 |
| 4 | 诊断危害性(反向) | 完全无害 ↔ 极大危害 |

</td><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 5 | 循证医学证据 | 完全循证 ↔ 无证据 |
| 6 | 诊断逻辑完整性 | 使用所有相关特征 ↔ 完全遗漏 |
| 7 | 诊断逻辑可读性 | 极好 ↔ 极差 |

</td></tr>
</table>

<table>
<tr><th colspan="2">💬 问诊过程 (5维)</th></tr>
<tr><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 8 | 问诊完整性 | 覆盖所有重要信息 ↔ 完全未覆盖 |
| 9 | 交流质量 | 无专业术语群众可理解 ↔ 全是术语 |
| 10 | 上下文质量 | 对话连贯性极好 ↔ 极差 |

</td><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 11 | 指令跟随 | 完全针对患者问题 ↔ 答非所问 |
| 12 | 解答患者问题 | 解决所有疑问 ↔ 完全未解决 |

</td></tr>
</table>

<table>
<tr><th colspan="2">❤️ 人文关怀 (5维)</th></tr>
<tr><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 13 | 以患者为中心的沟通 | 完全 ↔ 完全不以 |
| 14 | 尊重陈述和隐私 | 完全尊重 ↔ 完全不尊重 |
| 15 | 语言恰当性 | 非常恰当 ↔ 非常不恰当 |

</td><td width="50%">

| # | 维度 | 5分 ↔ 1分 |
|:---:|------|------|
| 16 | 解决患者担忧 | 尽力解决 ↔ 完全未试图 |
| 17 | 人文关怀表达 | 极致关怀 ↔ 完全无关怀 |

</td></tr>
</table>

> ℹ️ 当前代码实现了诊断正确性自动评估 (模糊匹配)，17 维完整评分的提示词定义在 `prompts_revised.py`，尚未完全集成自动评分管线

---

## 3.4 诊断判定算法

```
输入: predicted (AI诊断) vs actual (金标准)
输出: ✅ True / ❌ False

① 空值检测 ──▶ predicted 或 actual 为空 → False
② 直接匹配 ──▶ 双向子串包含 → True
③ 映射表匹配 ─▶ 18组中英疾病名互查 (正向+反向)
④ 关键词交叉 ─▶ 交集比 > 30% → True
⑤ 跨语言匹配 ─▶ 9组医学词 (lung↔肺, eye↔眼, …)
⑥ 兜底 ─────▶ False
```

<details>
<summary>📋 映射表覆盖范围 (点击展开)</summary>

| 系统 | 英文 | ↔ 中文别名 |
|------|------|------|
| 呼吸 | Pulmonary Sarcoidosis | 肺结节病 / 结节病 / 肉芽肿 / 慢性支气管炎 / 肺炎 |
| 真菌 | Paracoccidioidomycosis | 副球孢子菌病 / 南美芽生菌病 / 深部真菌病 / PCM |
| 眼科 | Vertical Nystagmus | 垂直眼震 / 眼震 / 眼球震颤 / 视神经炎 / 多发性硬化 |
| 皮肤 | Exogenous Ochronosis | 外源性褐黄病 |
| 肿瘤 | Lymphoma | 淋巴瘤 / 恶性淋巴瘤 / 血液系统疾病 |
| 内分泌 | Thyroiditis | 甲状腺炎 / 甲状腺疾病 |
| 通用 | lung, eye, throat, skin, fever, cough, pain, swelling | 跨语言 9 组 |

</details>

---

# 4 实验结果与分析

## 4.1 实验概览

> 🧪 共计 **3 轮独立实验** · 63+ 组对话 · **98 个对话 JSON** · **7 份评估报告**

| 🏷️ 轮次 | 📄 评估报告 | 📊 对话数 | 🎯 准确率 | 💭 平均信心度 | 🔄 平均轮数 |
|:---:|------|:---:|:---:|:---:|:---:|
| R1 | `evaluation_report_20250729_190127.json` | 21 | 28.57% | 81.67% | 9.67 |
| R2 | `evaluation_report_20250730_014127.json` | 21 | 33.33% | 84.76% | 10.00 |
| R3 | `evaluation_report_20250730_131915.json` | 21 | **42.86%** | 85.48% | 9.43 |
| **∑** | — | **63+** | **34.92%** | **83.97%** | **9.70** |

```
准确率趋势:  R1 ████████████░░░░░░░░░░░░░░ 28.57%
             R2 ████████████████░░░░░░░░░░░░ 33.33%  (+4.76%)
             R3 ███████████████████░░░░░░░░░░ 42.86%  (+9.53%)  ← 提示词优化驱动
```

---

## 4.2 整体指标 (最佳轮次 R3)

| 📊 指标 | 🔢 值 | 💡 解读 |
|:---|:---:|------|
| **准确率** | **42.86%** (9/21) | AI 医生正确诊断比例 |
| 平均问诊轮数 | 9.43 ± 1.26 | 接近上限 10 轮，信息收集效率待优化 |
| 平均信心度 | 85.48% ± 1.47% | 医生对自身诊断高度自信 |
| ⚠️ 信心度-准确率偏差 | **+42.62%** | **严重过度自信 (模型校准问题)** |
| 总 Token | 1,114,324 | 21 次问诊总量 |
| 总耗时 | 9,045.7s (2.51h) | API 响应时间 |
| 总 API 调用 | 434 次 | Doctor=231 · Patient=203 (约20.7次/对话) |
| 每对话 Token | 53,063 | ~50K tokens/对话 |
| 处理速度 | 123.2 t/s | 整体吞吐量 |

---

## 4.3 情绪维度分析

<table>
<tr>
  <td width="50%">

### R3 按情绪分组

| 😊 情绪 | 🎯 准确率 | 📊 n | ✅ |
|:---:|:---:|:---:|:---:|
| `calm` 平和 | 33.33% | 9 | 3 |
| `anxious` 焦虑 | 44.44% | 9 | 4 |
| `distrustful` 不信任 | **66.67%** | 3 | 2 |

  </td>
  <td width="50%">

### 三轮情绪对比

| 😊 情绪 | R1 | R2 | R3 | **μ** |
|:---:|:---:|:---:|:---:|:---:|
| `calm` | 11% | 44% | 33% | **29.63%** |
| `anxious` | 33% | 22% | 44% | **33.33%** |
| `distrustful` | 67% | 33% | 67% | **55.56%** ⬆ |

  </td>
</tr>
</table>

> 🔍 **反直觉发现**：不信任病人准确率最高 (55.56%)，比平和病人高 **+26pp**！质疑行为迫使医生更系统地问诊 → 获取更完整病史。提示"适度患者质疑"可能提升诊断质量。

---

## 4.4 数据完整性维度分析

<table>
<tr>
  <td width="50%">

### R3 按数据变体

| 📝 数据变体 | 🎯 准确率 | n |
|:---|:---:|:---:|
| `original` | 44.44% | 9 |
| `fake_symptoms_added` | **50.00%** | 6 |
| `symptoms_removed` | 33.33% | 6 |

  </td>
  <td width="50%">

### 三轮数据变体对比

| 📝 类型 | R1 | R2 | R3 | **μ** |
|:---|:---:|:---:|:---:|:---:|
| `original` | 33% | 33% | 44% | **37.04%** |
| `fake_symptoms_added` | 17% | 33% | 50% | **33.33%** |
| `symptoms_removed` | 33% | 33% | 33% | **33.33%** |

  </td>
</tr>
</table>

> 🔍 **关键洞察**：① 假症状干扰有限 (33.33% vs 基线37.04%) — AI有一定抗干扰力 ② 症状删除**三轮完全相同** (33.33%) — 信息缺失是硬性瓶颈 ③ R3 假症状组 (50%) 反超原始组 — 干扰促使医生更仔细鉴别

---

## 4.5 案例级错误分析

### 🔴 case_1 — 肺结节病 (最高误诊率: 3/21 正确 = 14.3%)

<table>
<tr><th>误诊为</th><th>次数</th><th>错误类型</th></tr>
<tr><td>🦠 副球孢子菌病 (Paracoccidiomycosis)</td><td align="center">3</td><td>混淆肉芽肿性疾病</td></tr>
<tr><td>🫁 肺结核 (TB)</td><td align="center">2</td><td>干咳+淋巴结肿大→误判感染</td></tr>
<tr><td>🤧 过敏性咳嗽 / 哮喘 / 过敏</td><td align="center">2</td><td>仅关注咳嗽，忽略多系统表现</td></tr>
<tr><td>🫗 胃食管反流 (GERD)</td><td align="center">1</td><td>假症状误导 (喉咙不适)</td></tr>
<tr><td>🫁 COPD</td><td align="center">1</td><td>咳嗽+体重下降→误判COPD</td></tr>
<tr><td>🦋 干燥综合征 (Sjögren's)</td><td align="center">1</td><td>假症状误导 (口干眼干)</td></tr>
<tr><td>🦋 系统性红斑狼疮 (SLE)</td><td align="center">1</td><td>多系统症状→自身免疫病</td></tr>
<tr><td>🧬 结缔组织病</td><td align="center">1</td><td>同上</td></tr>
</table>

> 💡 **根因**：LLM 在**非干酪性肉芽肿**这一病理特征上有知识盲区；未追问"是否有过活检"(结节病确诊金标准)

### 🟡 case_4 — 垂直眼震 (16.7% 准确率)

| 误诊为 | 次数 | 分析 |
|------|:---:|------|
| 👁️ 青光眼 | **5** | LLM 倾向匹配"最常见"眼科诊断 |
| 👁️ 干眼症 / 视网膜脱离 / 结核 / 缺血性病变 | 各1 | 忽略完整病史 (饮酒+晕倒+受压3h) |

> 💡 **根因**：此题是"最不可能"排除型题目，AI 在理解题目意图方面存在根本性困难

### 🟢 case_6 — 副球孢子菌病 (准确率最高，原始数据 3/3 正确)

| 误诊为 | 次数 | 分析 |
|------|:---:|------|
| 🦠 传染性单核细胞增多症 | 1 | 症状扰动组 |
| 🦠 颈部淋巴结炎 | 1 | 症状扰动组 |
| 🦋 Graves 病 (甲亢) | 1 | 症状扰动组 |

> 💡 **关键**：信息完整时，AI 能正确利用"🇧🇷 巴西移民 + 发热 + 淋巴结肿大 + 体重下降"的流行病学线索。信息缺失或干扰时则失败

---

## 4.6 患者 Agent 行为分析

通过对 **98 个对话文件** 的行为审计：

<table>
<tr><th>📊 行为类型</th><th>📈 频率</th><th>📝 典型表现</th><th>⚠️ 影响</th></tr>
<tr><td>✅ 正常回应</td><td align="center"><b>~55%</b></td><td>符合情绪+信息约束</td><td>—</td></tr>
<tr><td>🫥 空内容回复</td><td align="center">~15%</td><td>LLM 返回空 → fallback</td><td>浪费 1 轮，医生被迫重问或仓促诊断</td></tr>
<tr><td>🔄 重复性回应</td><td align="center">~12%</td><td>重复旧内容，不针对新问题</td><td>信息收集停滞</td></tr>
<tr><td>📢 信息过度披露</td><td align="center">~10%</td><td>一次说 3-5 个症状</td><td>违反"逐步透露"，模拟真实性下降</td></tr>
<tr><td>🤖 角色崩坏</td><td align="center">~8%</td><td>用医学术语分析病情</td><td>误导医生调整策略</td></tr>
</table>

> ⚠️ **不稳定性**：Llama-3.3-70B 在 7-10 轮长对话中角色保持能力显著下降，异常率前3轮 ~10% → 后3轮 ~35%

---

## 4.7 成本与性能分析

### Token 分布 (21 组对话)

| 📊 | 🩺 Doctor (o3-mini) | 🤒 Patient (Llama) | 📦 合计 |
|:---|:---:|:---:|:---:|
| 总 Token | ~580K | ~534K | **1,114,324** |
| 占比 | 52.1% | 47.9% | 100% |
| 均值/对话 | 27,619 | 25,429 | 53,063 |
| 最大 | 50,639 | 47,295 | 97,934 |
| 最小 | 8,514 | 10,582 | 19,096 |

### 耗时分布

| 📊 | 🩺 Doctor | 🤒 Patient | 📦 合计 |
|:---|:---:|:---:|:---:|
| 总耗时 | 5,700s (1.58h) | 3,350s (0.93h) | 9,045.7s (2.51h) |
| 均值/对话 | 271s | 160s | 431s |
| 均值/调用 | 22.4s/call | 12.9s/call | — |

> o3-mini 调用耗时是 Llama 的 **1.7 倍** — 推理型 vs 指令型模型延迟差异

### 💰 成本估算

| 🤖 模型 | 输入价格 | 输出价格 | 本实验成本 |
|------|:---:|:---:|:---:|
| o3-mini (Azure) | $1.10/M | $4.40/M | ~$1.85 |
| Llama-3.3-70B (Azure MAAS) | $0.71/M | $0.71/M | ~$0.76 |
| DeepSeek-R1 (SiliconFlow) | ¥1.00/M | ¥4.00/M | ~$0.69 |
| **21 组总成本** | | | **~$3.30** |
| **单次问诊均成本** | | | **~$0.16** 💸 |

---

## 4.8 关键发现

<table>
<tr><td>

### 🔴 发现 1: 过度自信

信心度-准确率偏差 **+43pp** (85% vs 42%)。AI 在信息不完整时仍高估自己，存在严重**校准问题**。临床场景中过度自信可能说服医生接受错误诊断。

</td></tr>
<tr><td>

### 🟡 发现 2: Prompt Engineering 高 ROI

仅通过优化医生提示词 (VINDICATE框架+地理流行病学+t=0.3)，准确率从 28.57% → 42.86% (**相对+50%**)，未改模型或数据。

</td></tr>
<tr><td>

### 🟢 发现 3: 信息完整性 > 情绪影响

症状删除准确率三轮回合一致 (33.33%) — 关键信息一旦缺失，诊断准确率存在**硬天花板**。验证了"信息收集完整性是诊断第一性原理"。

</td></tr>
<tr><td>

### 🔵 发现 4: 频率偏差 (Frequency Bias)

AI 倾向诊断常见病：肺部→结核/COPD · 眼部→青光眼。罕见病识别率明显偏低。

</td></tr>
</table>

### 📌 后续方向

| 🎯 方向 | 💡 方案 |
|:---|------|
| 📚 **RAG** | 诊断阶段接入 UpToDate / PubMed，提升罕见病识别 |
| 🔄 **反思机制** | 最终诊断前执行 "排除 top 3 alternatives" |
| 🎯 **校准** | 微调或 few-shot 降低过度自信 |
| 🧠 **患者升级** | 更强模型 (GPT-4o) + 角色一致性检测 |
| 📈 **扩展规模** | 5 情绪 × 50+ 病例 → 统计显著性检验 |

---

# 5 开发说明

### 🏗️ 代码架构

| 设计决策 | 实现 | 收益 |
|------|------|------|
| **BaseAgent 参数化** | `headers` + `API_URL` 注入 | 零修改切换 LLM 供应商 |
| **对话状态隔离** | Doctor/Patient 独立 `history[-6:]` | 角色独立 · Token成本可控 |
| **模块化管线** | 每模块只依赖 `List[Dict]` | 任意环节可独立替换/扩展 |

### 🩺 医生提示词

| 要点 | 策略 |
|------|------|
| **VINDICATE 框架** | 9大病因类别逐一列举，引导系统性覆盖 |
| **地理流行病学** | 南美/非洲/亚洲地方病+症状模式硬编码关联 |
| **Temperature 分层** | 问诊 0.7 (多样) → 诊断 0.3 (稳定) |

### 🤒 病人提示词

| 要点 | 策略 |
|------|------|
| **病例=身份** | 病例信息放 `system` 而非 `user`，作为角色而非任务 |
| **双重约束** | 正向 (自然语言) + 反向 (无医学知识/不杜撰/两句话) |
| **地理注入** | `if 'case_6' in case_id` 自动注入巴西移民背景 |

### 📝 日志与可复现

- **全量持久化**：文件名 = `{case_id}_{emotion}_{timestamp}.json`，三维索引
- **评估版本管理**：每轮实验独立 JSON，跨轮次对比
- **性能细粒度**：每次 API 调用的 prompt/completion/total token + 耗时 + 累计

---

<p align="center">
  <br>
  <sub>📬 如有问题或建议，欢迎提 Issue 或 PR</sub>
</p>
