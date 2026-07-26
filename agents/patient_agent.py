"""
Patient Agent - 模拟不同情绪状态的病人
"""
from typing import Dict, List, Optional
import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
# from configs.config import PATIENT_MODEL, PATIENT_EMOTIONS,OPENAI_4o,OPENAI_o3
from configs.config import PATIENT_MODEL, PATIENT_EMOTIONS,PATIENT_MODEL,LLAMA,API_URL_PATIENT

class PatientAgent(BaseAgent):
    """病人Agent，可以模拟不同情绪状态的病人"""
    
    def __init__(self, emotion_type: str = "calm", case_info: Dict = None):
        # super().__init__(PATIENT_MODEL, "patient",)
        print("病人：",PATIENT_MODEL,LLAMA,API_URL_PATIENT)
        super().__init__(PATIENT_MODEL,"patient",LLAMA,API_URL_PATIENT)
        self.emotion_type = emotion_type
        self.emotion_description = PATIENT_EMOTIONS.get(emotion_type, PATIENT_EMOTIONS["calm"])
        self.case_info = case_info or {}
        self.revealed_symptoms = set()  # 已经透露的症状
        self.all_symptoms = set()  # 所有症状
        self._initialize_symptoms()
    
    def _initialize_symptoms(self):
        """初始化症状信息"""
        if self.case_info:
            # 从病例信息中提取症状
            self.all_symptoms = set(self.case_info.get('symptoms', []))
            # 添加假症状（如果有）
            fake_symptoms = self.case_info.get('fake_symptoms', [])
            self.all_symptoms.update(fake_symptoms)
    
    def generate_response(self, doctor_question: str) -> str:
        """根据医生的问题生成回应"""
        # 构建系统提示词
        system_prompt = self._build_system_prompt()
        
        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史（保持正确的对话顺序）
        for msg in self.conversation_history[-6:]:  # 只保留最近6轮对话
            messages.append(msg)
        
        # 添加当前医生的问题
        messages.append({"role": "user", "content": f"医生问: {doctor_question}"})
        
        # 调用LLM生成回应
        response = self.call_llm(messages)
        
        # 检查响应是否为空或无效
        if not response or response.strip() == "":
            # 生成默认回应
            response = self._generate_fallback_response(doctor_question)
        
        # 记录对话
        self.add_to_history("user", f"医生问: {doctor_question}")
        self.add_to_history("assistant", response)
        
        return response
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        # 构建地理背景信息
        geographic_info = ""
        case_id = self.case_info.get('case_id', '')
        if 'case_6' in case_id:
            geographic_info = "\n- 地理背景：我最近从巴西移民过来，在那里生活了很多年。"
        elif self.case_info.get('source') == 'rare_disease':
            geographic_info = "\n- 地理背景：我有一些国际旅行或居住经历。"

        patient_info=f"""
        病例信息：
            - 主诉：{self.case_info.get('chief_complaint', '不适')}
            - 症状：{', '.join(self.all_symptoms)}
            - 病史：{self.case_info.get('history', '无特殊病史')}{geographic_info}
            """
        
        # base_prompt = f"""你是一个{self.emotion_description}病人。你正在向医生描述你的症状。
        base_prompt=f"""你是一个病人叫Tom，这是你的性格{self.emotion_description}，现在来医院就诊寻求疾病诊断帮助，这是你的病情：{patient_info}
        情绪特征："""
        
        # 根据不同情绪添加特定行为指导
        emotion_guides = {
            "calm": """
- 清晰、有条理地描述症状
- 配合医生的询问
- 主动提供相关信息""",
            
            "anxious": """
- 说话急促，经常打断医生
- 反复强调最担心的症状
- 询问是不是严重的疾病
- 表现出明显的担忧和恐惧""",
            
            "distrustful": """
- 对医生的问题表示怀疑
- 不愿意透露某些信息
- 质疑医生的专业性
- 要求解释每个问题的目的""",
            
            "confused": """
- 对症状的描述模糊不清
- 经常忘记或混淆信息
- 需要医生反复解释问题
- 可能提供矛盾的信息""",
            
            "aggressive": """
- 语气激动，可能有攻击性
- 抱怨等待时间长或医疗服务
- 要求立即得到诊断和治疗
- 可能威胁投诉或找其他医生"""
        }
        
        base_prompt += emotion_guides.get(self.emotion_type, emotion_guides["calm"])
        
        base_prompt +=f"""\n
        将有医生对你进行问诊，请使用自然的语言，回答医生的问到的问题，以便他了解你的疾病情况
        你没有医学知识，务必不要超出""{patient_info}""的描述，不要杜撰信息
        一次不要说出太多内容，不要说出医生没有问的其他症状
        回应规则：
        1. 保持角色的情绪特征
        2. 逐步透露症状，不要一次说完
        3. 根据医生的提问回答，不要偏离太远
        4. 使用口语化的表达方式
        5. 如果有假症状，也要自然地提及
        6. 不要阐述过多内容，保证在两句话内

        """
#         """

# 回应规则：
# 1. 保持角色的情绪特征
# 2. 逐步透露症状，不要一次说完
# 3. 根据医生的提问回答，不要偏离太远
# 4. 使用口语化的表达方式
# 5. 如果有假症状，也要自然地提及

# 请根据医生的问题，以病人的身份回应。"""

        
        # print(base_prompt)
        return base_prompt
    
    def set_fake_symptoms(self, fake_symptoms: List[str]):
        """设置假症状"""
        self.case_info['fake_symptoms'] = fake_symptoms
        self.all_symptoms.update(fake_symptoms)
    
    def remove_symptoms(self, symptoms_to_remove: List[str]):
        """移除某些症状（模拟信息缺失）"""
        for symptom in symptoms_to_remove:
            self.all_symptoms.discard(symptom)
    
    def _generate_fallback_response(self, doctor_question: str) -> str:
        """生成默认回应（当LLM返回空内容时）"""
        import random
        
        # 根据情绪类型生成不同的默认回应
        fallback_responses = {
            "calm": [
                "医生，您刚才问什么？我没太听清楚。",
                "请您再说一遍好吗？",
                "我想想...能再解释一下这个问题吗？"
            ],
            "anxious": [
                "医生，我好担心，您是说什么？",
                "不好意思，我太紧张了，没听清楚您的问题。",
                "您能再问一遍吗？我真的很担心我的情况。"
            ],
            "distrustful": [
                "您为什么要问这个？",
                "这个问题有什么意义吗？",
                "我不太明白您问这个的目的。"
            ]
        }
        
        responses = fallback_responses.get(self.emotion_type, fallback_responses["calm"])
        return random.choice(responses) 