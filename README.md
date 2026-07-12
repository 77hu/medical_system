<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Llama](https://img.shields.io/badge/LLM-Llama_3.3_70B-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://llama.meta.com/)
[![Claude](https://img.shields.io/badge/LLM-Claude_4.5-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Azure](https://img.shields.io/badge/Azure_AI-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)

</div>

<br/>

<h1 align="center">🏥 医疗 AI 多智能体诊断系统</h1>

<h3 align="center"><em>医生 Agent + 患者 Agent · 多 LLM 协同 · 症状生成 · 领域分类 · 诊断推理 · 诊疗计划</em></h3>

<br/>

---

## 📖 项目概述

基于大语言模型的多智能体医疗诊断系统，通过**医生 Agent** 和**患者 Agent** 的协同交互，实现症状分析、医学领域自动分类、诊断推理和诊疗计划生成。支持 **Llama 3.3 70B / Claude 4.5 / o3-mini** 多模型切换。

### 核心 Agent

| Agent | 角色 | 核心能力 |
|-------|------|---------|
| 🩺 医生 Agent | 诊断推理 | 症状分析 → 鉴别诊断 → 诊疗计划生成 |
| 🏥 患者 Agent | 症状描述 | 模拟真实患者陈述症状 + 病史 |

### Prompt 工程体系

| Prompt 文件 | 用途 |
|------|------|
| `医生agent.txt` | 医生诊断推理 + 诊疗计划 |
| `患者agent.txt` | 患者症状描述 + 病史 |
| `假症状规则.txt` | 合成数据生成规则 |
| `区分是诊断还是诊疗计划还是其他.txt` | 输出类型分类 |
| `区分病历是哪个领域.txt` | 医学领域自动分类 |

### 支持的 LLM

| 模型 | Provider | 端点 |
|------|----------|------|
| Llama 3.3 70B Instruct | Azure AI | `/models/chat/completions` |
| Claude 4.5 / 4.1 | Azure AI (Anthropic) | `/anthropic/v1/messages` |
| o3-mini | Azure OpenAI | `/openai/deployments/o3-mini` |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────┐
│          LLM Provider 层                  │
│  Azure AI + Azure OpenAI                 │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│          Agent 协同层                     │
│                                           │
│  患者 Agent ──→ 描述症状                 │
│       │                                   │
│       ▼                                   │
│  医生 Agent ──→ 诊断推理                │
│       │         ├── 领域分类              │
│       │         ├── 鉴别诊断              │
│       │         └── 诊疗计划              │
│       │                                   │
│       ▼                                   │
│  输出: 诊断 + 领域 + 诊疗计划 + 置信度    │
└──────────────────────────────────────────┘
```

---

## 🚀 快速开始

```bash
git clone https://github.com/77hu/medical_system.git
cd medical_system

pip install langchain langchain-openai anthropic python-dotenv

# 配置 Azure API Key
export AZURE_API_KEY="your-key-here"

# 运行 Agent
python doctor_agent.py
```

---

## 📁 项目结构

```
📦 medical_system/
├── 📜 医生agent.txt                         # 医生 Prompt
├── 📜 患者agent.txt                         # 患者 Prompt
├── 📜 假症状规则.txt                        # 合成数据规则
├── 📜 区分是诊断还是诊疗计划还是其他.txt       # 分类Prompt
├── 📜 区分病历是哪个领域.txt                  # 领域分类Prompt
└── 📘 README.md
```

---

## 📄 License

本项目仅供学习和研究使用。⚠️ 本系统仅为研究原型，不可用于真实医疗诊断。
