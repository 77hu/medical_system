<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![Llama](https://img.shields.io/badge/LLM-Llama_3.3_70B-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://llama.meta.com/)
[![Claude](https://img.shields.io/badge/LLM-Claude_4.5-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)

</div>

<br/>

<h1 align="center">🏥 医疗 AI 多智能体诊断系统</h1>

<h3 align="center"><em>医生 Agent · 患者 Agent · 症状生成 · 领域分类 · 诊断推理</em></h3>

<br/>

---

## 📑 目录

- [📖 项目概述](#-项目概述)
- [🏗️ 系统架构](#️-系统架构)
- [🧩 核心 Agent](#-核心-agent)
- [📋 Prompt 设计](#-prompt-设计)
- [🚀 快速开始](#-快速开始)

---

## 📖 项目概述

基于大语言模型的多智能体医疗诊断系统，通过医生 Agent 和患者 Agent 的协同交互，实现症状分析、领域分类、诊断推理和诊疗计划生成。

### 应用场景

| 场景 | 说明 |
|------|------|
| 症状分析 | 患者 Agent 描述症状，医生 Agent 分析 |
| 领域分类 | 自动识别病历所属医学领域 |
| 诊断推理 | 从症状到诊断的推理链路 |
| 假症状生成 | 合成数据增强，提升模型鲁棒性 |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────┐
│              LLM Provider 层              │
│  ├── Llama 3.3 70B (Azure AI)            │
│  ├── Claude 4.5 / 4.1 (Azure AI)         │
│  └── o3-mini (Azure OpenAI)              │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│            Agent 层                       │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │ 医生 Agent    │  │ 患者 Agent        │  │
│  │ · 诊断推理    │  │ · 症状描述         │  │
│  │ · 领域分类    │  │ · 病史提供         │  │
│  │ · 诊疗计划    │  │ · 假症状生成       │  │
│  └──────────────┘  └──────────────────┘  │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│            输出层                         │
│  诊断结果 · 诊疗计划 · 领域标签 · 置信度   │
└──────────────────────────────────────────┘
```

---

## 🧩 核心 Agent

### 医生 Agent

| 能力 | 说明 |
|------|------|
| 诊断推理 | 基于症状和病史推断疾病 |
| 诊疗计划 | 区分诊断/诊疗计划/其他 |
| 领域分类 | 识别病历所属医学专科 |
| 多模型支持 | Llama / Claude / o3-mini 可切换 |

### 患者 Agent

| 能力 | 说明 |
|------|------|
| 症状描述 | 模拟真实患者症状陈述 |
| 病史提供 | 生成相关的病史信息 |
| 假症状生成 | 合成低质量症状用于数据增强 |
| 真症状过滤 | 识别并去除不实症状 |

---

## 📋 Prompt 设计

| Prompt 文件 | 用途 |
|------|------|
| `医生agent.txt` | 医生诊断推理 + 诊疗计划生成 |
| `患者agent.txt` | 患者症状描述 + 病史陈述 |
| `假症状规则.txt` | 假症状生成规则与约束 |
| `区分是诊断还是诊疗计划还是其他.txt` | 输出类型分类 Prompt |
| `区分病历是哪个领域.txt` | 医学领域分类 Prompt |

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/77hu/medical_system.git
cd medical_system

# 2. 安装依赖
pip install langchain langchain-openai anthropic python-dotenv

# 3. 配置 API Keys（Azure AI / Azure OpenAI）
export AZURE_API_KEY="your-key-here"

# 4. 运行 Agent
python doctor_agent.py
```

---

## 📄 License

本项目仅供学习和研究使用。
