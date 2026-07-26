"""
配置文件 - 包含所有项目配置
敏感信息（API密钥）通过 .env 文件或环境变量加载
"""
import os


def _load_env_file():
    """从项目根目录的 .env 文件加载环境变量（若存在）"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

# ============================================
# API 配置 — 全部通过环境变量配置
# 复制 .env.example 为 .env 并填入真实密钥
# ============================================

# 评估模型 (SiliconFlow — DeepSeek)
API_KEY_EVAL = os.getenv('API_KEY_EVAL', '')
API_URL_EVAL = os.getenv('API_URL_EVAL', 'https://api.siliconflow.cn/v1/chat/completions')

# 医生模型 (Azure OpenAI — o3-mini)
API_KEY_DOCTOR = os.getenv('API_KEY_DOCTOR', '')
API_URL_DOCTOR = os.getenv('API_URL_DOCTOR', '')

# 病人模型 (Azure AI MAAS — Llama-3.3-70B)
API_KEY_PATIENT = os.getenv('API_KEY_PATIENT', '')
API_URL_PATIENT = os.getenv('API_URL_PATIENT', '')

# ============================================
# 模型配置
# ============================================

# 模型名称
DOCTOR_MODEL = os.getenv('DOCTOR_MODEL', 'gpt-o3mini')
PATIENT_MODEL = os.getenv('PATIENT_MODEL', 'Llama-3.3-70B-Instruct')
EVAL_MODEL = os.getenv('EVAL_MODEL', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

# 各厂商的 HTTP Headers（由 API_KEY 拼接）
OPENAI_o3 = {
    'Authorization': f'Bearer {API_KEY_DOCTOR}',
    'Content-Type': 'application/json'
}

LLAMA = {
    'Authorization': f'Bearer {API_KEY_PATIENT}',
    'Content-Type': 'application/json'
}

DEEPSEEK = {
    'Authorization': f'Bearer {API_KEY_EVAL}',
    'Content-Type': 'application/json'
}

# 本地开发时如需统一用 SiliconFlow 测试，取消以下注释：
# OPENAI_o3 = DEEPSEEK
# LLAMA = DEEPSEEK

# ============================================
# 病人情绪类型
# ============================================
PATIENT_EMOTIONS = {
    "calm": "平和的",
    "anxious": "焦虑的",
    "distrustful": "不信任的",
    "confused": "困惑的",
    "aggressive": "激动的"
}

# ============================================
# 数据路径
# ============================================
DATA_PATHS = {
    "rare_disease": "../data/rare_disease_302.json",
    "nejmai": "../data/nejmai_dataset.csv",
    "usmle_derm": "../data/usmle_and_derm_dataset.csv"
}

# ============================================
# 结果保存路径
# ============================================
RESULTS_DIR = "ai_patient_doctor_system/results"
CONVERSATIONS_DIR = os.path.join(RESULTS_DIR, "conversations")
EVALUATIONS_DIR = os.path.join(RESULTS_DIR, "evaluations")

# ============================================
# 对话参数
# ============================================
MAX_CONVERSATION_TURNS = 10  # 最大对话轮数
TEMPERATURE = 0.7            # 生成温度（医生诊断阶段内部使用0.3）

# ============================================
# 评估参数
# ============================================
EVALUATION_METRICS = ["accuracy", "f1_score", "precision", "recall"]

# ============================================
# 伪数据集保存路径
# ============================================
FAKE_DATA_PATHS = "data/generation_data/fake_data.json"
