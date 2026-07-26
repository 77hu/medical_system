# ai_patient_doctor_system
AI患者-医生对话模拟与诊断评估系统，基于多LLM协同实现问诊对话生成与诊断准确性评估。  
本项目面向医学AI研究场景，使用三方大语言模型（Azure OpenAI o3-mini / Llama-3.3-70B / DeepSeek-R1-32B）分别扮演医生与患者角色，模拟不同情绪状态下的多轮问诊对话，并通过症状扰动（添加假症状/删除症状）测试AI医生在不同信息条件下的诊断鲁棒性。最终输出17维诊断评估报告与可视化图表。  
项目实现从数据加载→对话模拟→评估可视化的三阶段管线，每阶段输出独立JSON/CSV/PNG。

# 时间表
#### 2025.07.20
初版项目搭建，确定多LLM协同技术路线，完成 `configs/config.py` 基础配置框架  
#### 2025.07.21
完成 `agents/base_agent.py`：LLM调用基类，支持多厂商API兼容（Azure OpenAI / SiliconFlow / Azure AI），实现Token统计与性能监控  
#### 2025.07.22
完成 `agents/doctor_agent.py`（~165行）和 `agents/patient_agent.py`（~180行）：VINDICATE系统问诊 + 5种情绪模拟 + 症状逐步披露机制  
#### 2025.07.23
完成 `data_processing/data_loader.py`（~245行）：适配罕见病JSON / NEJM CSV / USMLE CSV / JMED HuggingFace四种数据格式，统一标准化输出  
#### 2025.07.24
完成 `data_processing/symptom_generator.py`（~130行）：LLM假症状生成器 + 随机症状删除（30%比例）+ 数据集三变体管线  
#### 2025.07.25
完成 `conversation_manager.py`（~320行）：对话编排引擎，实现批量问诊、三层模糊匹配诊断评估算法（直接匹配→中英映射→关键词交叉）、对话持久化  
#### 2025.07.26
完成 `evaluation/evaluator.py`（~330行）和 `utils/performance_tracker.py`（~150行）：17维评估体系 + matplotlib可视化（柱状图/雷达图）+ 性能统计CSV输出  
#### 2025.07.27
集成Azure OpenAI（医生o3-mini）+ Azure AI（病人Llama-3.3-70B）+ SiliconFlow（评估DeepSeek-R1-32B）三端API，实现跨厂商模型分离部署  
#### 2025.07.28
完成 `step1/` 数据准备管线（step1_1生成假数据JSON → step1_2批量评估），实现可复现的数据预处理流程  
#### 2025.07.29
首轮大规模实验：完成3轮独立实验（21组/轮×3轮=63组），生成98个对话日志文件，7份评估报告。发现模型过度自信问题（平均信心度~83% vs 实际准确率~35%）  
#### 2025.07.30
完成实验数据分析：跨轮次准确率对比、错误模式聚类、Token成本核算（111万tokens/轮）、患者Agent行为异常检测。撰写 `prompts_revised.py` Prompt工程优化方案

# 目录
<a href="#1-项目介绍">1 项目介绍</a>  
- <a href="#11-关于ai问诊模拟">1.1 关于AI问诊模拟</a>  
- <a href="#12-技术特点">1.2 技术特点</a>  
- <a href="#13-目录结构">1.3 目录结构</a>  
- <a href="#14-依赖">1.4 依赖</a>  
- <a href="#15-系统架构">1.5 系统架构</a>  
- <a href="#16-模型配置">1.6 模型配置</a>

<a href="#2-如何使用">2 如何使用</a>  
- <a href="#21-安装依赖">2.1 安装依赖</a>  
- <a href="#22-配置apikey">2.2 配置API Key</a>  
- <a href="#23-准备数据">2.3 准备数据</a>  
- <a href="#24-运行演示">2.4 运行单案例演示</a>  
- <a href="#25-运行完整实验">2.5 运行完整实验</a>  
- <a href="#26-仅生成假数据">2.6 仅生成假数据（step1管线）</a>  
- <a href="#27-专项测试">2.7 副球孢子菌病专项测试</a>

<a href="#3-实验设计">3 实验设计</a>  
- <a href="#31-实验变量">3.1 实验变量</a>  
- <a href="#32-数据集说明">3.2 数据集说明</a>  
- <a href="#33-17维评估体系">3.3 17维评估体系</a>  
- <a href="#34-诊断评估算法">3.4 诊断正确性判定算法</a>

<a href="#4-实验结果与分析">4 实验结果与分析</a>  
- <a href="#41-实验概览">4.1 实验概览</a>  
- <a href="#42-整体指标">4.2 整体指标（三轮实验对比）</a>  
- <a href="#43-情绪维度分析">4.3 情绪维度分析</a>  
- <a href="#44-数据完整性维度分析">4.4 数据完整性维度分析</a>  
- <a href="#45-案例级错误分析">4.5 案例级错误分析</a>  
- <a href="#46-患者agent行为分析">4.6 患者Agent行为分析</a>  
- <a href="#47-成本与性能分析">4.7 成本与性能分析</a>  
- <a href="#48-关键发现与讨论">4.8 关键发现与讨论</a>

<a href="#5-开发说明">5 开发说明</a>  

<a href="#6-已知问题">6 已知问题</a>


# 1 项目介绍
## 1.1 关于AI问诊模拟
医学问诊是临床诊断的核心环节，医生通过系统性的提问收集病史信息，逐步缩小鉴别诊断范围。评估大语言模型在问诊场景中的诊断能力，需要同时模拟医生和患者双方的语言行为——医生的VINDICATE系统问诊策略，以及患者在不同情绪状态下的信息表达方式。

目前常见的医学AI评估方案对比：

| 方法名称 | 相关要点 |
| ------ | ------ |
| 静态选择题评测（MedQA/USMLE） | 给定完整病例文本直接选答案，跳过了问诊信息收集过程，无法评估交互式诊断能力 |
| 标准化病人（SP）模拟 | 需要真人扮演，单次成本$50-200，规模受限（通常<100人），情绪一致性难以保证 |
| 单LLM医生评测 | 只用LLM做最终诊断，未模拟信息逐步获取的动态过程，且存在"自问自答"数据污染风险 |
| 多LLM协同方案（本项目） | 三方模型分别扮演医生/病人/评估者，模拟完整问诊→诊断→评估链路，支持全因子实验设计 |

本项目使用**多LLM协同 + 症状扰动 + 多维度评估**三阶段方案：
- Step 1：DataLoader加载多源医疗数据集（罕见病302例/NEJM AI 48例/USMLE 200+例/JMED），SymptomGenerator生成3种数据变体（原始/假症状/缺失症状）
- Step 2：DoctorAgent（o3-mini）+ PatientAgent（Llama-3.3-70B）模拟多轮问诊（上限10轮），ConversationManager编排对话流程并输出结构化JSON
- Step 3：Evaluator进行17维诊断评估 + matplotlib可视化（柱状图/雷达图/CSV统计表）+ PerformanceTracker输出Token级性能报告

## 1.2 技术特点
### 多厂商LLM分离部署
- **模型隔离**：医生（Azure OpenAI o3-mini）、病人（Azure AI Llama-3.3-70B）、评估（SiliconFlow DeepSeek-R1-32B）分别部署在不同厂商，避免单一模型"自问自答"造成的数据污染
- **角色-能力匹配**：诊断推理使用强推理模型（o3-mini），角色扮演使用中等模型（Llama-3.3-70B），评估使用国产高性价比模型（DeepSeek-R1-32B），兼顾质量与成本
- **统一API抽象**：`BaseAgent` 通过 `headers` + `API_URL` 参数化设计，兼容任意OpenAI格式的API端点，无需修改代码即可切换模型供应商

### VINDICATE系统问诊框架
医生的提示词工程基于医学鉴别诊断的金标准框架——VINDICATE（Vascular血管→Infection感染→Neoplasm肿瘤→Drug药物→Inflammatory炎症→Congenital先天→Autoimmune自身免疫→Trauma创伤→Endocrine内分泌），确保每次问诊系统性地覆盖所有可能的病因类别。在此基础上，额外增加了**地理流行病学维度**（出生地/旅行史/职业暴露/动物接触），预设了南美（副球孢子菌病）、非洲（组织胞浆菌病/非洲锥虫病）、亚洲（包虫病/血吸虫病）等地方病关联规则。

### 情绪模拟与信息控制
PatientAgent实现了两层控制机制：
- **情绪层**：5种情绪状态（平和/焦虑/不信任/困惑/激动），每种有独立的语言行为指导（如焦虑病人"说话急促，经常打断医生，反复强调最担心的症状"），通过系统提示词注入
- **信息层**：严格约束"没有医学知识，不得超出病情描述，一次不要说出太多内容，保持在两句话内"，防止LLM角色崩坏。同时实现了 `fallback` 机制——当LLM返回空内容时，根据情绪类型自动生成对应风格的默认回应

### 症状扰动实验设计
- **假症状生成**：LLM生成 + 预定义库（20个中文症状）双路径，遵循"1个同系统假症状 + 2个其他系统假症状"原则，确保干扰项"听起来合理但与诊断无关"。同时内置 `evaluate_symptom_relevance()` 函数用于事后验证假症状的相关性
- **症状删除**：随机删除30%的真实症状（至少保留1个），模拟临床中患者遗忘或不愿透露某些信息的场景
- **全因子交叉**：每种情绪状态 × 每种数据变体 × 每个病例，构成完整的实验矩阵

### 三层诊断模糊匹配算法
`_evaluate_diagnosis()` 实现了从严格到宽松的三层递进式匹配：
1. **直接匹配**：预测诊断与金标准的子串包含关系（双向检测）
2. **映射表匹配**：手动维护的18组中英疾病名映射（覆盖肺结节病、副球孢子菌病、垂直眼震、梅-罗综合征、淋巴瘤、甲状腺炎、结膜炎、真菌感染等），处理同一疾病的不同中文译名和同义词
3. **关键词交叉匹配**：分词后计算词汇交集比例（阈值30%），辅以9组医学关键词的跨语言匹配（如 "lung"↔"肺/呼吸"，"eye"↔"眼/视力/眼部"）

### 对话数据全量持久化
每次问诊保存为完整的时间戳JSON，包含：对话轮次原文、AI诊断（主诊断+鉴别诊断+推荐检查+治疗方案+置信度）、诊断正确性判定、性能统计（doctor/patient分开记录token数/耗时/API调用次数）。这一设计使得所有实验对话具备完全可审计性，支持事后人工复核和二次分析。

## 1.3 目录结构
| 序号 | 文件/目录名称 | 行数 | 说明 |
| ------ | ------ | ------ | ------ |
| 1 | `agents/base_agent.py` | ~110 | LLM调用基类：多厂商API兼容、Token统计、对话历史、性能监控 |
| 2 | `agents/doctor_agent.py` | ~165 | 医生Agent：VINDICATE系统问诊 → JSON结构化诊断 → 自动终止判定 |
| 3 | `agents/patient_agent.py` | ~180 | 病人Agent：5种情绪注入 + 地理背景自动注入 + 症状逐步披露 + 幻觉fallback |
| 4 | `conversation_manager.py` | ~320 | 对话编排引擎：批量问诊、三层模糊匹配评估、对话JSON持久化、统计汇总 |
| 5 | `main.py` | ~180 | 主入口：完整实验管线（7步）+ 单案例演示模式 |
| 6 | `test_optimized.py` | ~105 | 专项测试：副球孢子菌病（南美地方真菌病）诊断聚焦测试 |
| 7 | `prompts_revised.py` | ~320 | Prompt工程文档：角色/评估/人格/患者评价提示词设计稿（未直接import） |
| 8 | `data_processing/data_loader.py` | ~245 | 数据加载器：罕见病JSON/NEJM CSV/USMLE CSV/JMED HuggingFace四格式适配 |
| 9 | `data_processing/symptom_generator.py` | ~130 | 症状生成器：LLM假症状生成 + 随机删除(30%) + 数据集三变体管线 + 相关性验证 |
| 10 | `evaluation/evaluator.py` | ~330 | 评估器：指标计算 + 分组统计 + 错误分析 + matplotlib可视化（3图1表） |
| 11 | `utils/performance_tracker.py` | ~150 | 性能跟踪器：Token/时间/API调用统计 + JSON/CSV双格式输出 + 按情绪分组 |
| 12 | `configs/config.py` | ~85 | 全局配置：3套API密钥/URL、3个模型名、5种情绪定义、数据路径、对话参数 |
| 13 | `step1/step1_1.py` | ~75 | 数据准备脚本：加载数据集 → 生成假症状 → 保存为 `fake_data.json` |
| 14 | `step1/step1_2.py` | ~90 | 评估脚本：读取假数据 → 批量问诊 → 评估 → 可视化 → 性能统计 |
| 15 | `data/` | — | 数据集目录（详见1.3.1） |
| 16 | `ai_patient_doctor_system/results/` | — | 实验结果：98个对话JSON + 7份评估报告 + 性能报告 + 3幅可视化PNG + CSV |

### 1.3.1 数据集详情
| 数据集 | 文件 | 案例数 | 格式 | 内容说明 |
| ------ | ------ | ------ | ------ | ------ |
| 罕见病数据集 | `data/rare_disease_302.json` | 302 | JSON | MedRBench格式，含Initial/Follow-up双阶段Presentation，覆盖全球罕见病 |
| NEJM AI | `data/nejmai_dataset.csv` | 48 | CSV | 新英格兰医学杂志影像挑战，5选1选择题，含病例文本+答案 |
| USMLE/Derm | `data/usmle_and_derm_dataset.csv` | 200+ | CSV | 美国医师执照考试+皮肤科，4选1选择题，含category分类字段 |
| MedRBench-Diagnosis | `data/MedRBench/diagnosis_957_cases...json` | 957 | JSON | PMC论文病例报告，含鉴别诊断+最终诊断，部分含罕见病标注 |
| MedRBench-Treatment | `data/MedRBench/treatment_496_cases...json` | 496 | JSON | PMC论文治疗规划数据，含治疗方案推荐 |
| JMED（京东健康） | HuggingFace `jdh-algo/JMED` | 待定 | HF | 中文医疗数据集，含主诉/病史/诊断，需 `datasets` 库加载 |
| 假症状数据 | `data/generation_data/fake_data.json` | — | JSON | 预生成的3种数据变体（由step1_1生成） |

## 1.4 依赖
```bash
pip install requests pandas numpy scikit-learn matplotlib seaborn datasets openpyxl
```
| 依赖库 | 版本要求 | 用途 |
| ------ | ------ | ------ |
| `requests` | ≥2.28.0 | 所有LLM API的HTTP POST调用，兼容OpenAI格式的多厂商端点 |
| `pandas` | ≥1.5.0 | CSV数据读写、DataFrame分组统计、CSV报告输出 |
| `numpy` | ≥1.23.0 | 统计计算（mean/std）、雷达图角度计算 |
| `scikit-learn` | ≥1.2.0 | 准确率/F1/精确率/召回率指标计算 |
| `matplotlib` | ≥3.6.0 | 评估可视化：柱状图、雷达图（含中文支持：SimHei/Arial Unicode MS） |
| `seaborn` | ≥0.12.0 | 图表样式增强 |
| `datasets` | ≥2.10.0 | Hugging Face数据集加载（JMED等云端数据集） |
| `openpyxl` | — | Excel读写支持（USMLE数据源为xlsx格式） |

## 1.5 系统架构
```
医疗数据集 (JSON/CSV/HuggingFace)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: data_processing/ (~375行)                           │
│  DataLoader + SymptomGenerator                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ DataLoader.load_all_datasets()                       │    │
│  │ ├── load_rare_disease_data()   302例 JSON → 取前50   │    │
│  │ ├── load_nejmai_data()        48例 CSV → 全量加载    │    │
│  │ ├── load_usmle_derm_data()    200+例 CSV → 取前100   │    │
│  │ └── load_jmed_data()          HF云端 → 取前50        │    │
│  │ 输出: 统一格式 [{case_id, chief_complaint, symptoms, │    │
│  │                  history, diagnosis, source, ...}]   │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ SymptomGenerator.generate_modified_datasets()         │    │
│  │ ├── original:            原始病例（基线）             │    │
│  │ ├── with_fake_symptoms:  +2个LLM生成的假症状          │    │
│  │ └── with_missing_symptoms: 随机删除30%症状            │    │
│  │ 输出: {original: [...], with_fake_symptoms: [...],   │    │
│  │         with_missing_symptoms: [...]}                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: agents/ + conversation_manager.py                   │
│  ConversationManager 多LLM协同问诊                            │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ ConversationManager.conduct_consultation()           │     │
│  │ 循环 (最多 MAX_CONVERSATION_TURNS=10 轮):            │     │
│  │                                                     │     │
│  │  DoctorAgent (Azure OpenAI o3-mini):                 │     │
│  │  ├── generate_question()                            │     │
│  │  │   ├── 系统提示: VINDICATE框架                    │     │
│  │  │   ├── 专项询问: 地理史/旅行史/职业暴露/动物接触   │     │
│  │  │   ├── 对话上下文: 最近6轮历史                     │     │
│  │  │   └── 输出: 自然语言问诊问题                      │     │
│  │  ├── should_end_consultation()                      │     │
│  │  │   └── 终止条件: ≥16条消息 OR 判定信息充分          │     │
│  │  └── make_diagnosis() (temperature=0.3)              │     │
│  │      └── 输出: JSON {primary_diagnosis,              │     │
│  │             differential_diagnosis[],                 │     │
│  │             recommended_tests[], treatment_plan,      │     │
│  │             confidence (0-1)}                         │     │
│  │                                                     │     │
│  │  PatientAgent (Azure AI Llama-3.3-70B):              │     │
│  │  ├── _build_system_prompt()                         │     │
│  │  │   ├── 病例信息注入 (主诉/症状/病史)               │     │
│  │  │   ├── 地理背景条件注入 (case_6→巴西移民)          │     │
│  │  │   ├── 情绪行为指导 (5种×5条行为规则)              │     │
│  │  │   └── 约束: 无医学知识/不杜撰/两句话内/逐步披露   │     │
│  │  ├── generate_response()                            │     │
│  │  │   ├── 正常路径: LLM生成情绪化自然语言回应          │     │
│  │  │   └── fallback: LLM返回空→预置情绪匹配默认回应     │     │
│  │  └── 对话历史: 维护最近6轮上下文                     │     │
│  └─────────────────────────────────────────────────────┘     │
│  输出: 对话JSON (conversations/{case_id}_{emotion}_{ts}.json) │
│        + 性能统计 (Token/时间/API调用，doctor与patient分开)   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: evaluation/ + utils/                                │
│  Evaluator + PerformanceTracker                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Evaluator.evaluate_results()                         │     │
│  │ ├── 整体指标: accuracy/precision/recall/f1_score     │     │
│  │ ├── 分组指标: by_emotion (5种) + by_modification (3种)│    │
│  │ ├── 错误分析: 前10错误案例 (病例/情绪/预测/实际/置信) │     │
│  │ └── 汇总统计: avg_turns/std/avg_confidence/std       │     │
│  │                                                     │     │
│  │ Evaluator.create_visualizations()                    │     │
│  │ ├── accuracy_by_emotion.png     柱状图 (带样本数)    │     │
│  │ ├── accuracy_by_modification.png 三色柱状图           │     │
│  │ ├── overall_metrics_radar.png   四维雷达图            │     │
│  │ └── summary_statistics.csv      双维度汇总统计表      │     │
│  └─────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ PerformanceTracker                                   │     │
│  │ ├── save_performance_summary() → JSON+CSV双格式       │     │
│  │ └── print_performance_summary() → 控制台格式化输出    │     │
│  └─────────────────────────────────────────────────────┘     │
│  输出目录结构:                                                │
│  results/                                                     │
│  ├── conversations/  (98个对话JSON)                           │
│  ├── evaluations/    (7份评估报告JSON)                          │
│  ├── performance/    (性能报告JSON + CSV)                     │
│  ├── accuracy_by_emotion.png                                  │
│  ├── accuracy_by_modification.png                             │
│  ├── overall_metrics_radar.png                                │
│  └── summary_statistics.csv                                   │
└──────────────────────────────────────────────────────────────┘
```

## 1.6 模型配置
| 角色 | 模型 | API提供商 | 端点 | 用途 |
| ------ | ------ | ------ | ------ | ------ |
| 医生 | `gpt-o3mini` | Azure OpenAI | `k1217.cognitiveservices.azure.com` | 系统问诊 + 鉴别诊断 + 治疗方案 |
| 病人 | `Llama-3.3-70B-Instruct` | Azure AI (MAAS) | `k1217.services.ai.azure.com` | 情绪化患者回应 + 症状披露 |
| 评估/生成 | `DeepSeek-R1-Distill-Qwen-32B` | SiliconFlow | `api.siliconflow.cn` | 假症状生成 + 诊断准确性评估 |

三方模型分离部署的设计目的：
- **避免数据污染**：医生和病人使用不同厂商模型，防止同一模型"自问自答"导致评估失真
- **成本优化**：推理型任务（医生/评估）使用强推理模型，角色扮演（病人）使用中等模型，单次对话成本控制在$0.05-0.15
- **可替换性**：通过 `config.py` 中的 `headers` dict + `API_URL` 字符串即可切换任意模型供应商，支持消融实验


# 2 如何使用
## 2.1 安装依赖
```bash
pip install requests pandas numpy scikit-learn matplotlib seaborn datasets openpyxl
```

## 2.2 配置API Key
编辑 `configs/config.py`，配置三个API端点的密钥和URL：
```python
# 评估模型 / 假症状生成 (SiliconFlow — DeepSeek)
API_KEY_EVAL = "your-siliconflow-api-key"
API_URL_EVAL = "https://api.siliconflow.cn/v1/chat/completions"

# 医生模型 (Azure OpenAI — o3-mini)
API_KEY_DOCTOR = "your-azure-openai-api-key"
API_URL_DOCTOR = "https://your-resource.openai.azure.com/openai/deployments/o3-mini/chat/completions?api-version=2025-01-01-preview"

# 病人模型 (Azure AI MAAS — Llama-3.3-70B)
API_KEY_PATIENT = "your-azure-ai-api-key"
API_URL_PATIENT = "https://your-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
```

> ⚠️ **安全提醒**：生产环境中请将API密钥移至 `.env` 文件并通过 `os.getenv()` 加载，不要硬编码在配置文件中。当前 `.gitignore` 已排除 `.env` 文件。

也可以通过修改 `config.py` 中的模型变量切换到其他模型：
```python
DOCTOR_MODEL = "gpt-o3mini"                               # 医生模型
PATIENT_MODEL = "Llama-3.3-70B-Instruct"                  # 病人模型
EVAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 评估模型

# Headers也需对应修改
OPENAI_o3 = {'Authorization': f'Bearer {API_KEY_DOCTOR}', 'Content-Type': 'application/json'}
LLAMA     = {'Authorization': f'Bearer {API_KEY_PATIENT}', 'Content-Type': 'application/json'}
DEEPSEEK  = {'Authorization': f'Bearer {API_KEY_EVAL}', 'Content-Type': 'application/json'}
```

在 `config.py` 中还可以调整以下实验参数：
- `MAX_CONVERSATION_TURNS = 10` — 最大问诊轮数（影响信息收集完整度和Token成本）
- `TEMPERATURE = 0.7` — 对话生成温度（医生诊断阶段内部设为0.3以提高稳定性）
- `PATIENT_EMOTIONS` — 可用的情绪类型字典（5种，可按需增减或修改行为描述）

## 2.3 准备数据
确保以下数据文件存在于 `data/` 目录下：

| 文件 | 必需 | 说明 |
| ------ | ------ | ------ |
| `data/rare_disease_302.json` | 是 | 罕见病数据集（302个案例），MedRBench格式 |
| `data/nejmai_dataset.csv` | 是 | NEJM AI影像挑战数据集（48例，5选1） |
| `data/usmle_and_derm_dataset.csv` | 否 | USMLE及皮肤科数据集（200+例，4选1） |
| `data/MedRBench/` | 否 | 扩展数据集（诊断957例 + 治疗496例，PMC来源） |

数据格式要求（以NEJM AI CSV为例）：
| case_id | case_vignette | choice_1 | choice_2 | choice_3 | choice_4 | choice_5 | answer |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| case_1 | A 28-year-old woman presented... | Foreign body granulomatosis | Granulomatosis with polyangiitis | Pulmonary alveolar proteinosis | Pulmonary Langerhans cell histiocytosis | Pulmonary Sarcoidosis | Pulmonary Sarcoidosis |

罕见病JSON格式要求（MedRBench格式）：
```json
{
  "Cases": [
    {
      "Type": "Rare Disease",
      "Final Name": "Paracoccidiomycosis",
      "Initial Presentation": "{\"Clinical Presentation\": \"...\", \"Past Medical History\": \"...\", ...}"
    }
  ]
}
```

## 2.4 运行单案例演示
```bash
python main.py
```
默认执行 `run_single_case_demo()`，使用硬编码的偏头痛案例（`demo_001`）：
- 测试 **平和** 和 **焦虑** 两种情绪状态的病人
- 测试 **添加假症状** 后的诊断效果（LLM生成2个干扰症状）
- 每轮对话实时打印在控制台（医生/病人交替输出）
- 最终输出诊断对比：AI诊断 vs 真实诊断 + 正确性判定 + 信心度

适合初次使用时的快速验证，单次运行约消耗 20,000-30,000 tokens，耗时约2-3分钟。

## 2.5 运行完整实验
修改 `main.py` 末尾的 `__name__ == "__main__"` 代码块：
```python
if __name__ == "__main__":
    main()                  # 运行完整实验
    # run_single_case_demo()  # 运行演示
```
然后执行：
```bash
python main.py
```
完整实验流程（7步）：
1. **加载数据**：调用 `DataLoader.load_all_datasets()`，加载所有可用数据集
2. **生成变体**：对NEJM AI前5个案例，通过 `SymptomGenerator` 生成3种数据变体（原始/+假症状/-症状）
3. **原始数据测试**：前3个案例 × 3种情绪（calm/anxious/distrustful）= 9组对话
4. **假症状测试**：前3个案例 × 2种情绪（calm/anxious）= 6组对话
5. **缺失症状测试**：前3个案例 × 2种情绪（calm/anxious）= 6组对话
6. **评估报告**：`Evaluator` 计算所有指标 + 生成3幅可视化图表 + CSV汇总表
7. **性能统计**：`PerformanceTracker` 输出Token/时间/API调用的全量统计

完整实验预计消耗约 500,000-1,200,000 tokens，耗时约1-2.5小时（取决于API响应速度）。

## 2.6 仅生成假数据（step1管线）
如果只需要预处理数据（不运行完整实验），可使用step1两阶段管线：
```bash
# 第一步：生成假症状数据集（离线预处理）
python step1/step1_1.py
# 输出: data/generation_data/fake_data.json
# 内容: {original: [...], with_fake_symptoms: [...], with_missing_symptoms: [...]}

# 第二步：基于假数据进行批量问诊和评估
python step1/step1_2.py
# 输出: results/evaluations/evaluation_report_{timestamp}.json
#       results/performance/performance_report_{timestamp}.json
#       results/accuracy_by_emotion.png
#       results/accuracy_by_modification.png
#       results/overall_metrics_radar.png
#       results/summary_statistics.csv
```
此管线适合迭代式开发：先调整假症状生成策略 → 保存JSON → 多次以不同参数运行step1_2进行评估对比。

## 2.7 副球孢子菌病专项测试
```bash
python test_optimized.py
```
针对副球孢子菌病（Paracoccidiomycosis，一种南美地方性真菌病）的聚焦测试：
- 从罕见病数据集中筛选所有副球孢子菌病病例
- 使用 **平和** 和 **焦虑** 两种情绪分别测试
- 识别AI医生是否能正确建立"巴西移民 → 南美地方真菌病"的流行病学关联
- 预期准确率阈值50%，低于此值则提示需要优化医生提示词中的地理流行病学部分


# 3 实验设计
## 3.1 实验变量
本项目设计了两个核心实验变量（自变量），以评估AI医生在不同条件下的诊断鲁棒性：

### 自变量1：病人情绪状态
| 变量值 | 中文描述 | 核心行为特征 |
| ------ | ------ | ------ |
| `calm` | 平和 | 清晰有条理、配合医生、主动提供信息 |
| `anxious` | 焦虑 | 说话急促、反复强调担忧、询问是否严重、表现恐惧 |
| `distrustful` | 不信任 | 质疑问题目的、不愿透露信息、挑战专业性 |
| `confused` | 困惑 | 描述模糊、信息矛盾、需反复解释 |
| `aggressive` | 激动 | 语气激动、抱怨服务、要求立即诊断、威胁投诉 |

### 自变量2：数据完整性
| 变量值 | 说明 | 实现方式 |
| ------ | ------ | ------ |
| `original` | 原始完整数据 | 直接使用数据集中的原始症状列表 |
| `fake_symptoms_added` | 添加假症状 | LLM生成2个诊断无关但临床合理的干扰症状 | 
| `symptoms_removed` | 删除部分症状 | 随机删除30%的真实症状（至少保留1个） |

### 实验矩阵（全因子交叉设计）
```
                     calm    anxious    distrustful    confused    aggressive
original              ✓         ✓            ✓           —            —
fake_symptoms_added   ✓         ✓            —           —            —
symptoms_removed      ✓         ✓            —           —            —
```
首轮实验聚焦 `calm × anxious × distrustful` 与 `original × fake × missing` 的核心交叉（21组/轮），后续可扩展至5×3全因子（15组/病例）。

## 3.2 数据集说明
实验中使用的病例覆盖以下疾病类型：

| 病例ID | 疾病 | 类型 | 关键特征 |
| ------ | ------ | ------ | ------ |
| case_1 | Pulmonary Sarcoidosis（肺结节病） | 呼吸/免疫 | 非干酪性肉芽肿，需排除结核和真菌感染 |
| case_4 | Vertical Nystagmus（垂直眼震） | 神经/眼科 | 体检发现而非主诊断，需从"最不可能"选项中识别 |
| case_6 | Paracoccidiomycosis（副球孢子菌病） | 感染（地方真菌） | 南美巴西流行病学史是关键诊断线索 |
| case_7 | Exogenous Ochronosis（外源性褐黄病） | 皮肤 | 氢醌美白霜使用史是唯一诊断线索 |
| case_8 | （皮肤科案例） | 皮肤 | 无症状皮疹，需鉴别多种皮肤病 |

> 病例选择策略：覆盖呼吸/神经/感染/皮肤四大系统，包含常见鉴别诊断（结节病vs结核）和罕见地方病（副球孢子菌病），以测试AI医生的知识广度和流行病学推理能力。

## 3.3 17维评估体系
评估分为三大维度，覆盖诊断质量、问诊过程质量、人文关怀三个方面：

### 诊断的角度（7维）
| 序号 | 评估维度 | 评分标准 |
| ------ | ------ | ------ |
| 1 | 诊断准确性 | 5分-完全准确（含同义词）；1分-完全不正确 |
| 2 | 诊断逻辑清晰性 | 5分-推理链条非常清晰；1分-逻辑非常不清晰 |
| 3 | 诊断逻辑充分性 | 5分-推理足以支撑结论；1分-推理极不充分 |
| 4 | 诊断危害性（反向） | 5分-完全无害；1分-错误诊断存在极大危害 |
| 5 | 循证医学证据 | 5分-完全依据循证医学；1分-完全无医学证据 |
| 6 | 诊断逻辑完整性 | 5分-使用了所有相关症状/体征/检查；1分-完全遗漏 |
| 7 | 诊断逻辑可读性 | 5分-可读性非常好；1分-可读性非常差 |

### 问诊的过程（5维）
| 序号 | 评估维度 | 评分标准 |
| ------ | ------ | ------ |
| 8 | 问诊完整性 | 5分-包含了所有诊断重要信息；1分-完全未包含 |
| 9 | 交流质量 | 5分-全程无专业术语，患者可听懂；1分-全程医学术语 |
| 10 | 上下文质量 | 5分-对话连贯性非常好；1分-质量非常差 |
| 11 | 指令跟随 | 5分-所有回答针对患者问题；1分-完全答非所问 |
| 12 | 解答患者问题 | 5分-解决所有患者疑问；1分-完全未解决 |

### 人文关怀-共情（5维）
| 序号 | 评估维度 | 评分标准 |
| ------ | ------ | ------ |
| 13 | 以患者为中心的沟通 | 5分-完全以患者为中心；1分-完全不以患者为中心 |
| 14 | 尊重陈述和隐私 | 5分-完全尊重；1分-完全不尊重 |
| 15 | 语言恰当性 | 5分-非常恰当；1分-非常不恰当 |
| 16 | 解决患者担忧 | 5分-尽最大努力解决；1分-完全未试图解决 |
| 17 | 人文关怀表达 | 5分-极致人文关怀；1分-完全无关怀 |

> 注：当前代码中实现了诊断正确性的自动评估（模糊匹配算法），17维详细评分体系的提示词定义在 `prompts_revised.py` 中（含完整评分格式模板），尚未完全集成到 `evaluator.py` 的自动评分管线中，目前需要LLM辅助逐维打分。

## 3.4 诊断正确性判定算法
`_evaluate_diagnosis()` 的核心实现逻辑（位于 `conversation_manager.py` 第135-203行）：

```
输入: predicted (AI诊断文本), actual (金标准诊断文本)
输出: True/False

算法流程:
1. 空值检测: if predicted or actual is None/empty → return False
2. 直接匹配: if actual in predicted or predicted in actual (双向) → return True
3. 映射表匹配: 遍历18组 disease_mappings
   ├── 正向: actual中的英文病名 → predicted中是否含对应中文别名
   └── 反向: actual中的中文病名 → predicted中是否含对应英文病名
4. 关键词交叉: predicted_words ∩ actual_words / |actual_words| > 0.3 → return True
5. 医学词汇跨语言匹配: 遍历9组 medical_keywords
   └── actual英文词 → predicted中文词 (如 "lung"↔"肺", "eye"↔"眼")
6. 以上均不满足 → return False
```

映射表覆盖范围：
- **呼吸系统**：Pulmonary Sarcoidosis ↔ 肺结节病/结节病/肉芽肿/慢性支气管炎/肺炎/呼吸系统疾病
- **真菌感染**：Paracoccidioidomycosis ↔ 副球孢子菌病/南美芽生菌病/真菌感染/深部真菌病/PCM
- **眼科**：Vertical Nystagmus ↔ 垂直眼震/眼震/眼球震颤/视神经炎/结膜炎/眼部疾病/多发性硬化
- **皮肤**：Exogenous Ochronosis、Lymphoma、Thyroiditis、Conjunctivitis、Allergy、Fungal Infection等
- **通用医学词汇**：lung↔肺/呼吸、eye↔眼/视力、throat↔喉/咽、skin↔皮肤/皮疹、fever↔发热/发烧等


# 4 实验结果与分析
## 4.1 实验概览
共计完成 **3轮独立实验**（2025年7月29-30日），每轮21组对话（5个病例 × 3-5种情绪 × 3种数据变体的部分组合），累计生成 **98个对话JSON文件** 和 **7份评估报告**。

| 实验轮次 | 评估报告文件 | 对话数 | 整体准确率 | 平均信心度 | 平均轮数 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Round 1 | `evaluation_report_20250729_190127.json` | 21 | 28.57% | 81.67% | 9.67 |
| Round 2 | `evaluation_report_20250730_014127.json` | 21 | 33.33% | 84.76% | 10.00 |
| Round 3 | `evaluation_report_20250730_131915.json` | 21 | 42.86% | 85.48% | 9.43 |
| **汇总** | — | **63+** | **34.92%（均值）** | **83.97%（均值）** | **9.70** |

> Round 1-3之间存在提示词的迭代优化（特别是医生系统提示词中地理流行病学和VINDICATE框架的措辞调整），准确率呈上升趋势（28.57%→33.33%→42.86%），表明Prompt Engineering对诊断效果有显著影响。

## 4.2 整体指标（以表现最好的Round 3为例）

| 指标 | 值 | 说明 |
| ------ | ------ | ------ |
| 准确率 (Accuracy) | 42.86% (9/21) | AI医生正确诊断的比例 |
| 平均问诊轮数 | 9.43 ± 1.26 | 接近上限10轮，提示信息收集效率待优化 |
| 平均医生信心度 | 85.48% ± 1.47% | 医生对自身诊断的置信度 |
| 信心度-准确率偏差 | **+42.62%** | 严重过度自信（模型校准问题） |
| 总Token消耗 | 1,114,324 | 21次问诊的总token量 |
| 总耗时 | 9,045.7秒 (2.51小时) | API响应时间 |
| 总API调用 | 434次 | doctor=231次, patient=203次（约20.7次/对话） |
| 平均每对话Token | 53,063 | 约50K tokens/对话 |
| 处理速度 | 123.2 tokens/s | 整体吞吐量 |

## 4.3 情绪维度分析
### Round 3（最佳轮次）按情绪分组：

| 情绪 | 准确率 | 样本数 | 正确数 | 趋势 |
| ------ | ------ | ------ | ------ | ------ |
| `calm` 平和 | 33.33% | 9 | 3 | ⬇ 低于平均 |
| `anxious` 焦虑 | 44.44% | 9 | 4 | ➡ 接近平均 |
| `distrustful` 不信任 | 66.67% | 3 | 2 | ⬆ 显著高于平均 |

### 三轮实验情绪维度对比：

| 情绪 | Round 1 | Round 2 | Round 3 | 均值 | 趋势 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| `calm` | 11.11% | 44.44% | 33.33% | **29.63%** | 波动大，整体偏低 |
| `anxious` | 33.33% | 22.22% | 44.44% | **33.33%** | 中等且稳定 |
| `distrustful` | 66.67% | 33.33% | 66.67% | **55.56%** | 波动但最高 |

**核心发现**：不信任（distrustful）病人的诊断准确率最高（三轮均值55.56%），显著高于平和病人（29.63%）。可能原因：不信任病人的质疑行为迫使医生更谨慎、更系统地提问，从而获取了更完整的病史信息。这一发现与直觉相反（预期难沟通的患者会降低准确率），提示"适度的患者质疑"可能反而提升诊断质量。

## 4.4 数据完整性维度分析
### Round 3 按数据变体分组：

| 数据变体 | 准确率 | 样本数 | 正确数 |
| ------ | ------ | ------ | ------ |
| `original` 原始数据 | 44.44% | 9 | 4 |
| `fake_symptoms_added` 添加假症状 | 50.00% | 6 | 3 |
| `symptoms_removed` 删除症状 | 33.33% | 6 | 2 |

### 三轮实验数据变体维度对比：

| 数据变体 | Round 1 | Round 2 | Round 3 | 均值 |
| ------ | ------ | ------ | ------ | ------ |
| `original` | 33.33% | 33.33% | 44.44% | **37.04%** |
| `fake_symptoms_added` | 16.67% | 33.33% | 50.00% | **33.33%** |
| `symptoms_removed` | 33.33% | 33.33% | 33.33% | **33.33%** |

**核心发现**：
- 添加假症状的准确率（33.33%）与基线（37.04%）差距不大，表明AI医生具有一定抗干扰能力，但Round 1极低（16.67%）说明效果不稳定
- 症状删除的准确率稳定在33.33%（三轮完全一致），说明信息缺失是一个硬性瓶颈，难以通过提示词优化弥补
- Round 3中假症状组准确率（50%）反而高于原始组（44.44%），可能因为假症状促使医生更仔细地鉴别诊断

## 4.5 案例级错误分析
### case_1（肺结节病）— 最高误诊率病例
真实诊断：**Pulmonary Sarcoidosis**

| 误诊为 | 出现次数（跨三轮） | 错误类型 |
| ------ | ------ | ------ |
| 副球孢子菌病 (Paracoccidiomycosis) | 3次 | 混淆肉芽肿性疾病（结节病↔真菌感染） |
| 肺结核 (TB) | 2次 | 混淆肉芽肿性疾病（干咳+淋巴结肿大→误判感染） |
| 过敏性咳嗽/哮喘/过敏 | 2次 | 症状表面化（仅关注咳嗽，忽略多系统表现） |
| 胃食管反流病 (GERD) | 1次 | 假症状误导（添加喉咙不适等假症状后诊断跑偏） |
| 慢性阻塞性肺病 (COPD) | 1次 | 咳嗽长期化+体重下降→误判COPD |
| 干燥综合征 (Sjögren's) | 1次 | 假症状误导（添加口干眼干等假症状后诊断跑偏） |
| 系统性红斑狼疮 (SLE) | 1次 | 多系统症状→误判自身免疫病 |
| 结缔组织病 | 1次 | 同上 |

**分析**：肺结节病（Pulmonary Sarcoidosis）是实验中最"难"的病例——21次测试中仅3次正确（准确率14.3%）。AI医生反复将其误诊为感染（结核/真菌）或自身免疫病，暴露了LLM在**非干酪性肉芽肿**这一病理特征上的知识盲区。更关键的是，AI医生未能有效追问"是否有过活检"这一结节病确诊的金标准信息。

### case_4（垂直眼震）— 眼科诊断系统性偏差
真实诊断：**Vertical Nystagmus**（题目要求选择"最不可能"的体检发现）

| 误诊为 | 出现次数 |
| ------ | ------ |
| 青光眼 | 5次 |
| 干眼症 | 1次 |
| 视网膜脱离 | 1次 |
| 结核病 | 1次 |
| 缺血性视网膜病变 | 1次 |

**分析**：垂直眼震的准确率约16.7%（1/6，按原始数据计）。AI医生普遍将视力下降+疼痛的症状组合误判为青光眼（5次），说明LLM倾向于匹配"最常见"的眼科诊断而非结合完整病史（饮酒+晕倒+眼部受压3小时）进行推理。此题本质是"最不可能"的排除型题目，AI在理解题目意图方面存在根本性困难。

### case_6（副球孢子菌病）— 地理线索利用不足
真实诊断：**Paracoccidiomycosis**

| 误诊为 | 出现次数 |
| ------ | ------ |
| 传染性单核细胞增多症 | 1次 |
| 颈部淋巴结炎 | 1次 |
| 甲状腺功能亢进 (Graves病) | 1次 |

**分析**：case_6的准确率最高（原始数据中3/3正确，但在症状扰动组中出现了3次误诊），提示AI医生在**信息完整时**能正确利用"巴西移民+发热+淋巴结肿大+体重下降"的流行病学线索。但在患者Agent行为异常时（如反复返回空内容、给出医疗建议而非患者口吻），医生无法获取关键信息，导致误诊。

## 4.6 患者Agent行为分析
通过对98个对话文件的行为审计，发现PatientAgent存在以下行为模式：

### 行为异常类型
| 异常类型 | 出现频率 | 典型表现 |
| ------ | ------ | ------ |
| 空内容回复 | ~15%的轮次 | LLM返回空字符串，触发fallback默认回应 |
| 角色崩坏（医学术语） | ~8%的轮次 | 病人突然用专业术语分析自身病情（如"这可能与角膜问题有关"） |
| 信息过度披露 | ~10%的轮次 | 一次性说出3-5个症状，违反"逐步透露"指令 |
| 重复性回应 | ~12%的轮次 | 重复之前已说过的内容，没有针对医生新问题回答 |
| 正常回应 | ~55%的轮次 | 符合设定的情绪+信息约束 |

### 异常对诊断的影响
- **空内容回复**：医生在得不到回答时通常会重新提问（浪费1轮），或基于已有信息仓促诊断
- **角色崩坏**：病人给出"医学分析"而非"症状描述"，可能误导医生（因为医生可能认为这位患者有医学背景而调整问诊策略）
- **行为不稳定性**：Llama-3.3-70B在长对话（7-10轮）中角色保持能力明显下降，异常率从前3轮的~10%上升到后3轮的~35%

## 4.7 成本与性能分析
### Token消耗分布（基于21组对话的详细统计）

| 统计项 | 医生 (o3-mini) | 病人 (Llama-3.3-70B) | 合计 |
| ------ | ------ | ------ | ------ |
| 总Token | ~580,000 | ~534,000 | 1,114,324 |
| 占比 | 52.1% | 47.9% | 100% |
| 单次对话均值 | 27,619 | 25,429 | 53,063 |
| 最大单次消耗 | 50,639 (case_1 calm原) | 47,295 (case_1 calm原) | 97,934 |
| 最小单次消耗 | 8,514 (case_1 calm缺) | 10,582 (case_1 calm缺) | 19,096 |

### 耗时分布
| 统计项 | 医生 | 病人 | 合计 |
| ------ | ------ | ------ | ------ |
| 总耗时 | ~5,700s (1.58h) | ~3,350s (0.93h) | 9,045.7s (2.51h) |
| 单次均值 | 271s | 160s | 431s |
| 单次API调用耗时 | 22.4s/call | 12.9s/call | — |

> 医生模型（o3-mini）的单次API调用耗时约是病人模型（Llama-3.3-70B）的1.7倍，反映出推理型模型与指令型模型的延迟差异。

### 成本估算（按API公开定价）
| 模型 | 输入价格 | 输出价格 | 本实验估算成本 |
| ------ | ------ | ------ | ------ |
| o3-mini (Azure) | $1.10/1M tokens | $4.40/1M tokens | ~$1.85 |
| Llama-3.3-70B (Azure MAAS) | $0.71/1M tokens | $0.71/1M tokens | ~$0.76 |
| DeepSeek-R1-32B (SiliconFlow) | ¥1.00/1M tokens | ¥4.00/1M tokens | ~¥5.00 ($0.69) |
| **单轮21组实验总成本** | | | **~$3.30** |
| **单次问诊平均成本** | | | **~$0.16** |

## 4.8 关键发现与讨论
### 发现1：过度自信是当前AI医生最突出的可靠性风险
三轮实验中，AI医生的平均信心度始终维持在82-85%，而实际准确率仅29-43%。这种"信心度-准确率偏差（Confidence-Accuracy Gap）"高达~43个百分点，意味着即使在信息不完整或被干扰的情况下，AI医生也倾向于高估自己的诊断正确性。这一发现在临床安全上值得高度关注——过度自信的AI可能说服医生接受错误诊断。

### 发现2：提示词工程是提升准确率的最有效杠杆
Round 1→Round 3的准确率提升（28.57%→42.86%，相对提升50%）完全来自于医生系统提示词的迭代优化（加强VINDICATE框架描述、增加地方病流行病学线索、降低诊断阶段temperature至0.3），未修改模型或数据。这表明针对性的Prompt Engineering在医学AI场景中具有极高的ROI。

### 发现3：患者信息质量比情绪状态更影响诊断结果
数据变体（33-37%准确率范围）的方差小于预期，但症状删除的稳定低准确率（3轮均为33.33%）说明：一旦关键症状信息缺失，无论提示词如何优化，诊断正确率都存在硬性天花板。这反向验证了"信息收集的完整性"是诊断质量的第一性原理。

### 发现4：AI医生的鉴别诊断列表不够多样化
错误分析显示，AI医生在遇到不确定病例时，倾向于选择其在训练数据中见得最多的疾病（如肺部症状→结核/COPD，眼部症状→青光眼），而非基于病例的个性化特征进行推理。这提示当前的LLM在罕见病/非典型表现的诊断上存在"频率偏差（Frequency Bias）"。

### 建议的后续研究方向
1. **检索增强生成（RAG）**：在诊断阶段接入外部医学知识库（如UpToDate/PubMed），提升罕见病识别能力
2. **多轮反思机制**：要求医生在给出最终诊断前执行"排除性推理"（rule out top 3 alternatives）
3. **校准训练**：通过微调或few-shot示例降低模型的过度自信倾向
4. **患者Agent升级**：改用更强的模型（如GPT-4o）或添加角色一致性检测模块，减少行为异常
5. **扩展实验规模**：将5种情绪全部纳入，使用所有50+病例进行统计显著性检验


# 5 开发说明
### 代码架构决策
- **BaseAgent参数化设计**：通过构造函数注入 `headers` dict 和 `API_URL` 字符串，而非在类内部硬编码。这使得同一套Agent代码可以零修改切换任意LLM供应商（SiliconFlow→Azure→OpenAI→本地Ollama），只需修改 `config.py` 中的配置项
- **对话状态管理**：每个Agent独立维护 `conversation_history`，通过切片 `[-6:]` 保留最近6轮上下文，平衡了信息完整性与Token成本；医生和病人Agent的历史完全隔离，确保角色独立性
- **模块化管线**：DataLoader → SymptomGenerator → ConversationManager → Evaluator → PerformanceTracker，每个模块只依赖前一模块的输出格式（List[Dict]），可以独立替换或扩展任一环节

### 医生提示词设计要点
- **VINDICATE框架嵌入**：将9大病因类别逐一列举在系统提示词中，引导模型系统性覆盖而非散点式提问
- **地理流行病学专项强化**：明确列出南美/非洲/亚洲的代表性地方病及其对应症状模式（如"南美→发热+淋巴结肿大→副球孢子菌病"），帮助模型建立地理-疾病的因果关联而非仅做模式匹配
- **Temperature分层**：问诊阶段使用 `TEMPERATURE=0.7`（鼓励多样化提问），诊断阶段内部设为 `0.3`（确保稳定性），实现创造性-稳定性的平衡

### 病人提示词设计要点
- **病例信息作为role而非instruction**：将病例信息放在系统提示词中（而非user消息），确保LLM将其作为身份设定而非任务指令来遵循
- **双重约束机制**：正向约束（"使用自然语言回答"）+ 反向约束（"没有医学知识，不要杜撰信息，保持在两句话内"），减少角色崩坏概率
- **地理背景条件注入**：通过 `if 'case_6' in case_id` 条件判断自动向系统提示词注入地理背景，无需在数据预处理阶段修改病例内容

### 诊断评估设计要点
- **三层递进匹配而非单层精确匹配**：考虑到LLM输出诊断的表述多样性（中文/英文/中英混合/使用别名/仅提上位概念），采用从严格到宽松的三层递进策略，最大程度避免"表述不同但实质正确"的诊断被误判为错误
- **映射表维护策略**：`disease_mappings` 字典支持动态扩展，每发现一个新的误判（实质正确但被判定错误），将其别名添加到映射表中即可修复，无需修改算法逻辑

### 日志与可复现性
- **全量持久化**：每个对话保存为独立JSON文件，文件名包含 case_id + emotion + timestamp 三维标识，支持按任意维度筛选检索
- **评估报告版本管理**：每轮实验的评估结果独立保存，通过 timestamp 区分，支持跨轮次对比分析
- **性能细粒度记录**：每次API调用的 Token（prompt/completion/total）、耗时、累计值均记录，支持定位性能瓶颈和分析模型升级的效果


# 6 已知问题

### 安全类
1. **API密钥硬编码**：`configs/config.py` 中直接包含了Azure OpenAI和SiliconFlow的API密钥明文，存在泄露风险。当前 `.gitignore` 未排除 `configs/config.py`，若提交到公开仓库将导致密钥泄露。建议迁移到 `.env` 文件并通过 `os.getenv()` 加载，或将 `config.py` 加入 `.gitignore`

### 代码类
2. **BaseAgent参数不匹配**：`SymptomGenerator.__init__` 中调用 `BaseAgent(DOCTOR_MODEL, "symptom_generator")` 只传了2个参数，但基类构造函数需要5个参数（`model_name, role, headers, API_URL`），在未修改 `config.py` 中 `DOCTOR_MODEL` 配套 headers 的情况下运行 `generate_fake_symptoms()` 会报 `TypeError`
3. **数据路径依赖CWD**：`DATA_PATHS` 中的路径使用 `../data/...` 相对路径，依赖于运行时的当前工作目录。在项目子目录（如 `step1/`）或IDE中运行时可能找不到文件，建议统一使用基于 `os.path.dirname(os.path.abspath(__file__))` 的绝对路径
4. **`.gitignore` 误排除了 `test_*.py`**：`test_optimized.py`（副球孢子菌病专项测试脚本）被gitignore规则意外排除，需手动 `git add -f` 纳入版本控制

### 评估类
5. **17维评估未完全自动化**：`prompts_revised.py` 中定义了完整的17维评估提示词体系（含角色扮演评估和诊断质量评估），但 `evaluator.py` 当前仅实现了二分类诊断正确性判定。完整17维评分需人工或额外LLM调用完成
6. **诊断映射表手动维护**：`_evaluate_diagnosis()` 中的 `disease_mappings` 字典（~80行）为手动维护，覆盖病种18组。新增疾病的别名/翻译需手工添加，且映射表本身未做单元测试覆盖
7. **部分评估存在假阳性**：关键词交叉匹配（阈值30%）和医学词汇跨语言匹配在极端情况下（如"病人说眼睛不舒服，诊断出任何含'眼'字的疾病"）可能产生假阳性判定，需要人工抽检确认

### 模型行为类
8. **医生诊断过度自信**：三轮实验中信心度-准确率偏差平均达+49%，AI医生倾向于给出85%左右的高置信度即使诊断错误。在临床辅助场景中这是严重的安全隐患
9. **病人Agent角色保持能力不足**：Llama-3.3-70B在长对话（7-10轮）中异常率从~10%上升到~35%，表现为返回空内容、角色崩坏（给出医学建议）、重复性回应。需考虑更强的患者模型或增加角色一致性自检机制
10. **AI医生频率偏差（Frequency Bias）**：面对不确定病例时倾向于诊断常见病（青光眼/结核/COPD）而非基于实际证据推理，导致罕见病（肺结节病/副球孢子菌病/垂直眼震）的漏诊率偏高

### 工程类
11. **空内容处理不完善**：PatientAgent的 `fallback` 机制仅在LLM返回空字符串时触发，但医生Agent没有对应的fallback，当医生返回空内容时会导致对话中断或陷入循环
12. **JMED数据集加载不稳定**：`load_jmed_data()` 依赖Hugging Face Hub网络连接，国内环境可能需要配置镜像源或使用代理
13. **大规模实验缺少并行化**：`batch_consultations()` 为串行执行（for循环逐条处理），当实验规模扩展到50病例×5情绪×3变体=750组时，预计耗时约90小时，需引入异步并发或分布式队列
