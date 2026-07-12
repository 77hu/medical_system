# medical_system
AI多智能体医患对话模拟系统，基于LLM实现医生-患者智能体自主问诊。  
本项目针对医学AI研究场景，使用DeepSeek大模型驱动Doctor Agent和Patient Agent两个独立智能体，模拟真实医患交互过程。支持5种患者情绪模拟、虚假症状注入、信息缺失实验，用于研究不同患者行为对诊断准确率的影响。  
项目已生成~70个完整对话记录（JSON格式），覆盖罕见病、NEJM AI挑战赛、USMLE等多个医学数据集。

# 时间表
#### 2025.03.09
初版项目搭建，确定LLM多智能体技术路线  
#### 2025.03.10
完成BaseAgent抽象基类开发，统一LLM API调用和Token追踪  
#### 2025.03.12
完成DoctorAgent开发，实现VINDICATE鉴别诊断原则  
#### 2025.03.14
完成PatientAgent开发，实现5种情绪和症状渐进披露  
#### 2025.03.15
开发SymptomGenerator模块，实现虚假症状注入和信息缺失实验  
#### 2025.03.16
完成ConversationManager对话编排引擎开发  
#### 2025.03.18
完成Evaluator评估模块，生成准确率柱状图和雷达图  
#### 2025.03.20
完成PerformanceTracker性能追踪模块  
#### 2025.03.22
集成4个医学数据集（罕见病302例/NEJM/USMLE/MedRBench）  
#### 2025.03.25
完成批量实验，生成~70个对话记录和评估报告

# 目录
<a href="#1-项目介绍">1 项目介绍</a>  
- <a href="#关于ai医患对话模拟">1.1 关于AI医患对话模拟</a>  
- <a href="#目录结构">1.2 目录结构</a>  
- <a href="#依赖">1.3 依赖</a>  
- <a href="#系统架构">1.4 系统架构</a>  

<a href="#如何使用">2 如何使用</a>  
- <a href="#安装依赖">2.1 安装依赖</a>  
- <a href="#配置api">2.2 配置API</a>  
- <a href="#运行对话实验">2.3 运行对话实验</a>  
- <a href="#查看结果">2.4 查看结果</a>  
- <a href="#数据集说明">2.5 数据集说明</a>  

<a href="#统计数据">3 统计数据</a>  
- <a href="#实验产出">3.1 实验产出</a>  

<a href="#开发说明">4 开发说明</a>  

<a href="#已知问题">5 已知问题</a>  


# 1 项目介绍
## 1.1 关于AI医患对话模拟
传统医疗诊断研究依赖真实医患对话数据，获取困难且隐私保护要求高。AI多智能体系统可以生成大量合成对话数据，用于研究不同患者行为（情绪、信息完整性）对诊断质量的影响。

目前常见的方案对比：

| 方法名称 | 相关要点 |
| ------ | ------ |
| 真实医患对话分析 | 数据质量高但获取困难，隐私保护要求严格 |
| 规则式对话系统 | 可控但不够灵活，无法模拟复杂患者行为 |
| 单LLM角色扮演 | 简单但缺乏智能体间真实交互 |
| 多智能体对话 | 本项目采用的方案，两个独立LLM模拟真实医患交互 |

本项目使用**DeepSeek多智能体 + 情绪控制 + 症状修改实验**方案，支持完整的研究管线：
- Doctor Agent：遵循VINDICATE原则，逐轮提问和结构化诊断
- Patient Agent：5种情绪（平静/焦虑/不信任/困惑/攻击性）
- SymptomGenerator：虚假症状注入 + 信息缺失实验
- Evaluator：多维度评估（准确率/精确率/召回率/F1）
- PerformanceTracker：Token/时间/调用次数统计

## 1.2 目录结构
### 1.2.1 核心代码
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | `main.py` | 入口脚本 |
| 2 | `conversation_manager.py` | 对话编排主模块 |
| 3 | `agents/base_agent.py` | 基础Agent（LLM交互抽象） |
| 4 | `agents/doctor_agent.py` | 医生Agent（VINDICATE诊断） |
| 5 | `agents/patient_agent.py` | 患者Agent（5种情绪+症状策略） |
| 6 | `agents/data_processing/data_loader.py` | 数据加载器 |
| 7 | `agents/data_processing/symptom_generator.py` | 症状修改生成器 |
| 8 | `evaluation/evaluator.py` | 诊断准确率评估+可视化 |
| 9 | `utils/performance_tracker.py` | Token/时间/调用统计 |
| 10 | `configs/config.py` | API密钥、模型、路径配置 |

### 1.2.2 数据与结果
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | `data/rare_disease_302.json` | 罕见病数据集（302例） |
| 2 | `data/MedRBench/` | MedRBench基准（957+496例） |
| 3 | `data/nejmai_dataset.csv` | NEJM AI挑战赛数据集 |
| 4 | `data/usmle_and_derm_dataset.csv` | USMLE皮肤科数据集 |
| 5 | `results/conversations/*.json` | 对话记录（~70个） |
| 6 | `results/evaluations/*.json` | 评估报告 |
| 7 | `results/performance/*.json/.csv` | 性能指标 |

## 1.3 依赖
```
pip install requests pandas numpy matplotlib seaborn scikit-learn
```
LLM API：SiliconFlow（SiliconCloud，OpenAI兼容端点）

## 1.4 系统架构
```
┌──────────────────┐           ┌──────────────────┐
│   Patient Agent  │◄─────────►│   Doctor Agent   │
│  - 5种情绪        │  对话交互   │  - VINDICATE原则  │
│  - 症状渐进披露    │          │  - 结构化诊断      │
│  - 虚假症状注入    │          │  - 置信度评分      │
└────────┬─────────┘           └────────┬─────────┘
         └──────────┬───────────────────┘
                    ▼
           Conversation Manager (对话编排)
                    ▼
       ┌────────────┼─────────────────┐
       ▼            ▼                 ▼
   DataLoader   Evaluator    PerformanceTracker
   (数据加载)   (评估+可视化)  (Token/时间统计)
```

# 2 如何使用
## 2.1 安装依赖
```
pip install requests pandas numpy matplotlib seaborn scikit-learn
```

## 2.2 配置API
在`configs/config.py`中配置SiliconFlow API：
```python
API_KEY = "your-api-key"
DOCTOR_MODEL = "deepseek-r1-distill-qwen-32b"
PATIENT_MODEL = "deepseek-r1-distill-qwen-14b"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
```

## 2.3 运行对话实验
```
python main.py
```
或运行测试脚本：
```
python test_optimized.py   # 副球孢子菌病案例测试
```

## 2.4 查看结果
- 对话记录：`results/conversations/*.json`（完整JSON对话）
- 评估报告：`results/evaluations/*.json`（按情绪/修改类型分组）
- 性能统计：`results/performance/*.json`（Token/时间/调用数）

## 2.5 数据集说明
| 数据集 | 格式 | 描述 |
| ------ | ------ | ------ |
| rare_disease_302.json | JSON | 302例罕见病（主诉+既往病史） |
| MedRBench | JSON | 诊断957例+治疗496例基准 |
| nejmai_dataset.csv | CSV | NEJM AI挑战赛数据 |
| usmle_and_derm_dataset.csv | CSV | USMLE皮肤科实践题 |

# 3 统计数据
## 3.1 实验产出
| 项目 | 数量 |
| ------ | ------ |
| 对话记录 | ~70个JSON文件 |
| 评估报告 | 多个（按情绪/修改类型分组） |
| 数据集 | 4个（罕见病/MedRBench/NEJM/USMLE） |
| 情绪类型 | 5种（平静/焦虑/不信任/困惑/攻击性） |
| 症状修改 | 3版本（原始/虚假/缺失） |

# 4 开发说明
- Doctor模型temperature=0.3（诊断需要稳定性），Patient模型temperature=0.7（对话需要多样性）
- 诊断匹配使用字符串相似度 + 同义词映射字典（中英双语疾病名）
- VINDICATE原则：血管性、炎症/感染、肿瘤、退行性、医源性/中毒、先天性、自身免疫性、创伤性、内分泌/代谢

# 5 已知问题
1. API Key存储在`LLM_key.txt`和`config.py`中，建议移至环境变量
2. 历史实验诊断准确率为0%（跨语言匹配问题，同义词映射不完整）
3. 部分路径使用中文字符，跨平台可能有编码问题
4. 数据集包含中英双语，诊断匹配需要完善同义词词典
