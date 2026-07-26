"""
Doctor Agent - 模拟医生进行问诊和诊断
"""
from typing import Dict, List, Optional, Tuple
import json
import re
import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from configs.config import DOCTOR_MODEL,OPENAI_o3,API_URL_DOCTOR


class DoctorAgent(BaseAgent):
    """医生Agent，负责问诊和诊断"""
    
    def __init__(self):
        # super().__init__(DOCTOR_MODEL, "doctor")
        super().__init__(DOCTOR_MODEL,'doctor',OPENAI_o3,API_URL_DOCTOR)
        self.diagnosis = None
        self.treatment_plan = None
        self.collected_symptoms = []
        self.differential_diagnosis = []
    
    def generate_question(self, patient_response: str = None) -> str:
        """生成下一个问诊问题"""
        system_prompt = """你是一位经验丰富的医生，正在进行问诊。你需要：
1. 系统地收集病史和症状信息，遵循VINDICATE原则
2. **重点关注地理史和流行病学因素**
3. 根据病人的回答调整问诊策略
4. 注意病人的情绪状态并适当回应
5. 逐步缩小诊断范围

问诊原则：
- 从主诉开始，逐步深入
- **必须询问地理史**：出生地、居住史、旅行史（特别是热带/亚热带地区）
- 询问症状的性质、部位、时间、诱因等
- 了解既往史、家族史、用药史、职业史、动物接触史
- 考虑感染性疾病（包括真菌、寄生虫等）
- 保持专业但友善的态度

特别注意：
- 对于发热+淋巴结肿大+体重下降的病人，要考虑感染（包括真菌感染）、肿瘤、自身免疫性疾病
- 南美背景的病人要考虑副球孢子菌病等地方性真菌感染
- 询问是否来自疫区或有相关暴露史

请生成下一个问诊问题。"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        
        if patient_response:
            messages.append({"role": "user", "content": f"病人说: {patient_response}"})
        else:
            messages.append({"role": "user", "content": "开始问诊，请询问病人的主诉。"})
        
        response = self.call_llm(messages)
        self.add_to_history("assistant", response)
        
        return response
    
    def make_diagnosis(self) -> Dict[str, any]:
        """根据收集的信息做出诊断"""
        system_prompt = """你是一位经验丰富的医生，需要根据问诊收集的信息做出诊断。

诊断思路（VINDICATE原则）：
- V：血管性疾病
- I：感染性疾病（细菌、病毒、真菌、寄生虫）
- N：肿瘤性疾病
- D：药物性/中毒
- I：炎症性/免疫性疾病
- C：先天性疾病
- A：自身免疫性疾病
- T：创伤性疾病
- E：内分泌代谢性疾病

重点关注流行病学因素：
- **地理史**：患者的出生地、居住史、旅行史
- **职业暴露史**和动物接触史
- **当地流行疾病**

常见地理相关疾病：
- 南美/中美：副球孢子菌病(Paracoccidiomycosis)、美洲锥虫病
- 非洲：非洲锥虫病、组织胞浆菌病
- 亚洲：包虫病、血吸虫病等

对于发热+淋巴结肿大+体重下降，特别考虑：
1. 感染性疾病：真菌感染（副球孢子菌病、组织胞浆菌病）、结核病、病毒感染
2. 恶性肿瘤：淋巴瘤、白血病、转移性肿瘤
3. 自身免疫性疾病：SLE、类风湿性关节炎等

请分析对话历史，并提供：
1. 最可能的诊断（考虑地理因素）
2. 鉴别诊断（其他可能的诊断）
3. 建议的进一步检查
4. 初步治疗方案

请以JSON格式输出：
{
    "primary_diagnosis": "主要诊断",
    "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"],
    "recommended_tests": ["检查1", "检查2"],
    "treatment_plan": "治疗方案描述",
    "confidence": 0.85  // 诊断信心度，0-1之间
}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请根据以下问诊记录做出诊断：\n" + self._format_conversation_for_diagnosis()}
        ]
        
        response = self.call_llm(messages, temperature=0.3)  # 降低温度以获得更稳定的诊断
        
        # 解析JSON响应
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                diagnosis_data = json.loads(json_match.group())
                self.diagnosis = diagnosis_data
                return diagnosis_data
            else:
                # 如果解析失败，返回默认结构
                return {
                    "primary_diagnosis": "需要进一步检查",
                    "differential_diagnosis": [],
                    "recommended_tests": ["血常规", "生化检查"],
                    "treatment_plan": "暂时对症治疗，等待检查结果",
                    "confidence": 0.3
                }
        except Exception as e:
            print(f"诊断解析错误: {e}")
            return {
                "primary_diagnosis": "解析错误",
                "differential_diagnosis": [],
                "recommended_tests": [],
                "treatment_plan": "需要重新评估",
                "confidence": 0.0
            }
    
    def _format_conversation_for_diagnosis(self) -> str:
        """格式化对话历史用于诊断"""
        formatted = []
        for i, msg in enumerate(self.conversation_history):
            role = "医生" if msg["role"] == "assistant" else "病人"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def should_end_consultation(self) -> bool:
        """判断是否应该结束问诊"""
        # 如果已经问了足够多的问题，或者收集了足够的信息
        if len(self.conversation_history) >= 16:  # 8轮对话
            return True
        
        # 检查最近的对话是否表明已经收集了足够信息
        if len(self.conversation_history) >= 4:
            recent_response = self.conversation_history[-1]["content"]
            if "足够" in recent_response or "可以诊断" in recent_response:
                return True
        
        return False 