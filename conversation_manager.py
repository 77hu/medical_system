"""
对话管理器 - 管理医生和病人之间的对话流程
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import sys
sys.path.append('.')
from agents.patient_agent import PatientAgent
from agents.doctor_agent import DoctorAgent
from configs.config import MAX_CONVERSATION_TURNS, CONVERSATIONS_DIR


class ConversationManager:
    """管理医生和病人之间的对话"""
    
    def __init__(self):
        self.conversations = []
        self.current_conversation = None
        
        # 确保对话保存目录存在
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    
    def conduct_consultation(self, 
                           case_info: Dict, 
                           patient_emotion: str = "calm",
                           save_conversation: bool = True) -> Dict:
        """进行一次完整的问诊对话"""
        
        # 初始化agents
        patient = PatientAgent(emotion_type=patient_emotion, case_info=case_info)
        doctor = DoctorAgent()
        
        # 记录对话元数据
        conversation_data = {
            'case_id': case_info.get('case_id', 'unknown'),
            'patient_emotion': patient_emotion,
            'true_diagnosis': case_info.get('diagnosis', ''),
            'modification_type': case_info.get('modification_type', 'original'),
            'start_time': datetime.now().isoformat(),
            'turns': []
        }
        
        # 开始对话
        print(f"\n{'='*80}")
        print(f"🏥 开始问诊")
        print(f"   📋 病例ID: {case_info.get('case_id')}")
        print(f"   😊 病人情绪: {patient_emotion}")
        print(f"   🩺 真实诊断: {case_info.get('diagnosis', '未知')}")
        print(f"   🔄 数据类型: {case_info.get('modification_type', 'original')}")
        print(f"{'='*80}")
        
        # 医生的第一个问题
        print(f"\n💬 第1轮对话:")
        doctor_question = doctor.generate_question()
        print(f"👨‍⚕️ 医生: {doctor_question}")
        
        turn_count = 0
        while turn_count < MAX_CONVERSATION_TURNS:
            # 病人回答
            patient_response = patient.generate_response(doctor_question)
            print(f"🤒 病人: {patient_response}")
            
            # 记录这一轮对话
            conversation_data['turns'].append({
                'turn': turn_count + 1,
                'doctor': doctor_question,
                'patient': patient_response
            })
            
            # 检查是否应该结束问诊
            if doctor.should_end_consultation():
                print(f"\n✅ 问诊结束 (共 {turn_count + 1} 轮对话)")
                break
            
            # 医生继续提问
            turn_count += 1
            print(f"\n💬 第{turn_count + 1}轮对话:")
            doctor_question = doctor.generate_question(patient_response)
            print(f"👨‍⚕️ 医生: {doctor_question}")
            
        if turn_count >= MAX_CONVERSATION_TURNS - 1:
            print(f"\n⏰ 达到最大对话轮数 ({MAX_CONVERSATION_TURNS}) 自动结束")
        
        # 医生做出诊断
        print(f"\n{'🔍 诊断分析':=^80}")
        diagnosis = doctor.make_diagnosis()
        
        # 记录诊断结果
        conversation_data['end_time'] = datetime.now().isoformat()
        conversation_data['doctor_diagnosis'] = diagnosis
        conversation_data['total_turns'] = len(conversation_data['turns'])
        
        # 评估诊断准确性
        is_correct = self._evaluate_diagnosis(
            diagnosis['primary_diagnosis'], 
            case_info.get('diagnosis', '')
        )
        conversation_data['diagnosis_correct'] = is_correct
        
        # 收集性能统计
        doctor_stats = doctor.get_performance_stats()
        patient_stats = patient.get_performance_stats()
        conversation_data['performance_stats'] = {
            'doctor': doctor_stats,
            'patient': patient_stats
        }
        
        print(f"\n📋 诊断结果:")
        print(f"   🩺 AI诊断: {diagnosis['primary_diagnosis']}")
        print(f"   ✅ 真实诊断: {case_info.get('diagnosis', '')}")
        print(f"   {'✅ 正确' if is_correct else '❌ 错误'}")
        print(f"   🎯 信心度: {diagnosis.get('confidence', 0):.1%}")
        
        print(f"\n📊 性能统计:")
        total_tokens = doctor_stats['total_tokens'] + patient_stats['total_tokens']
        total_time = doctor_stats['total_time'] + patient_stats['total_time']
        total_calls = doctor_stats['api_calls'] + patient_stats['api_calls']
        print(f"   🔢 总Token数: {total_tokens:,}")
        print(f"   ⏱️  总耗时: {total_time:.2f}s")
        print(f"   📞 API调用: {total_calls} 次")
        print(f"   🚀 处理速度: {total_tokens/max(total_time, 0.001):.1f} tokens/s")
        
        # 保存对话记录
        if save_conversation:
            filepath = self._save_conversation(conversation_data)
            print(f"\n💾 对话已保存: {filepath}")
        
        self.conversations.append(conversation_data)
        self.current_conversation = conversation_data
        
        return conversation_data
    
    def _evaluate_diagnosis(self, predicted: str, actual: str) -> bool:
        """评估诊断是否正确"""
        if not predicted or not actual:
            return False
            
        predicted_lower = predicted.lower()
        actual_lower = actual.lower()
        
        # 疾病名称映射表（英文到中文，以及常见的同义词）
        disease_mappings = {
            'pulmonary sarcoidosis': ['肺结节病', '结节病', '肺部结节病', '肺肉芽肿', '慢性支气管炎', '支气管炎', '肺部疾病', '肺炎', '呼吸系统疾病'],
            'sarcoidosis': ['结节病', '肺结节病', '肺部结节病', '肉芽肿', '慢性炎症', '炎症性疾病'],
            'paracoccidiomycosis': ['副球孢子菌病', '副球孢子菌', '南美芽生菌病', '真菌感染', '深部真菌病', '真菌病', '南美真菌病', 'PCM', '球孢子菌病', '巴西芽生菌病'],
            'paracoccidioidomycosis': ['副球孢子菌病', '副球孢子菌', '南美芽生菌病', '真菌感染', '深部真菌病', '真菌病', '南美真菌病', 'PCM', '球孢子菌病', '巴西芽生菌病'],
            'vertical nystagmus': ['垂直眼震', '眼震', '眼球震颤', '垂直性眼震', '视神经炎', '结膜炎', '眼部疾病', '视力问题', '眼部炎症', '角膜', '眼压', '多发性硬化'],
            'melkersson-rosenthal syndrome': ['梅-罗综合征', '面神经麻痹', '唇肿胀', '面部肿胀', '神经系统疾病'],
            'lymphoma': ['淋巴瘤', '恶性淋巴瘤', '淋巴系统肿瘤', '淋巴结肿大', '血液系统疾病'],
            'thyroiditis': ['甲状腺炎', '甲状腺疾病', '甲状腺问题', '内分泌疾病'],
            'thyroid': ['甲状腺', '甲状腺疾病', '甲状腺问题', '甲状腺结节', '甲状腺功能'],
            'conjunctivitis': ['结膜炎', '眼部炎症', '红眼病', '眼部感染'],
            'allergy': ['过敏', '过敏反应', '过敏性疾病', '过敏性炎症'],
            'infection': ['感染', '炎症', '感染性疾病', '细菌感染', '病毒感染'],
            'fungal infection': ['真菌感染', '深部真菌病', '真菌病', '霉菌感染'],
            'mycosis': ['真菌病', '真菌感染', '深部真菌感染', '霉菌病']
        }
        
        # 直接匹配
        if actual_lower in predicted_lower or predicted_lower in actual_lower:
            return True
        
        # 检查疾病映射
        for eng_disease, chinese_variants in disease_mappings.items():
            if eng_disease in actual_lower:
                for chinese_variant in chinese_variants:
                    if chinese_variant in predicted_lower:
                        return True
            
            # 反向检查
            for chinese_variant in chinese_variants:
                if chinese_variant in actual_lower and eng_disease in predicted_lower:
                    return True
        
        # 关键词匹配（降低阈值到30%）
        predicted_words = set(predicted_lower.split())
        actual_words = set(actual_lower.split())
        
        if len(predicted_words & actual_words) / max(len(actual_words), 1) > 0.3:
            return True
        
        # 医学相关词汇的模糊匹配
        medical_keywords = {
            'lung': ['肺', '呼吸'],
            'eye': ['眼', '视力', '眼部'],
            'throat': ['喉', '咽', '喉咙'],
            'neck': ['颈', '脖子', '颈部'],
            'skin': ['皮肤', '皮疹', '疹子'],
            'fever': ['发热', '发烧'],
            'cough': ['咳嗽', '咳'],
            'pain': ['疼痛', '痛', '疼'],
            'swelling': ['肿胀', '肿', '水肿']
        }
        
        for eng_word, chinese_words in medical_keywords.items():
            if eng_word in actual_lower:
                for chinese_word in chinese_words:
                    if chinese_word in predicted_lower:
                        return True
        
        return False
    
    def _save_conversation(self, conversation_data: Dict) -> str:
        """保存对话记录到文件"""
        filename = f"conversation_{conversation_data['case_id']}_{conversation_data['patient_emotion']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def batch_consultations(self, 
                          cases: List[Dict], 
                          emotions: List[str] = None,
                          save_all: bool = True) -> List[Dict]:
        """批量进行问诊"""
        if emotions is None:
            emotions = ["calm"]
        
        all_results = []
        total_cases = len(cases) * len(emotions)
        current = 0
        
        for case in cases:
            for emotion in emotions:
                current += 1
                print(f"\n\n{'🔄 批量实验进度':=^80}")
                print(f"📊 进度: {current}/{total_cases}")
                print(f"📋 当前病例: {case.get('case_id', 'unknown')}")
                print(f"😊 情绪状态: {emotion}")
                print(f"{'='*80}")
                
                try:
                    result = self.conduct_consultation(
                        case_info=case,
                        patient_emotion=emotion,
                        save_conversation=save_all
                    )
                    all_results.append(result)
                    print(f"✅ 病例 {case.get('case_id')} ({emotion}) 完成")
                except Exception as e:
                    print(f"❌ 处理病例 {case.get('case_id')} ({emotion}) 时出错: {e}")
                    continue
        
        return all_results
    
    def get_statistics(self, results: List[Dict] = None) -> Dict:
        """获取对话统计信息"""
        if results is None:
            results = self.conversations
        
        if not results:
            return {}
        
        stats = {
            'total_conversations': len(results),
            'by_emotion': {},
            'by_modification': {},
            'overall_accuracy': 0,
            'average_turns': 0,
            'average_confidence': 0
        }
        
        correct_count = 0
        total_turns = 0
        total_confidence = 0
        
        for conv in results:
            # 统计准确率
            if conv.get('diagnosis_correct'):
                correct_count += 1
            
            # 统计轮数
            total_turns += conv.get('total_turns', 0)
            
            # 统计信心度
            if conv.get('doctor_diagnosis'):
                total_confidence += conv['doctor_diagnosis'].get('confidence', 0)
            
            # 按情绪分组
            emotion = conv.get('patient_emotion', 'unknown')
            if emotion not in stats['by_emotion']:
                stats['by_emotion'][emotion] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0
                }
            stats['by_emotion'][emotion]['total'] += 1
            if conv.get('diagnosis_correct'):
                stats['by_emotion'][emotion]['correct'] += 1
            
            # 按修改类型分组
            mod_type = conv.get('modification_type', 'original')
            if mod_type not in stats['by_modification']:
                stats['by_modification'][mod_type] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0
                }
            stats['by_modification'][mod_type]['total'] += 1
            if conv.get('diagnosis_correct'):
                stats['by_modification'][mod_type]['correct'] += 1
        
        # 计算平均值和准确率
        stats['overall_accuracy'] = correct_count / len(results) if results else 0
        stats['average_turns'] = total_turns / len(results) if results else 0
        stats['average_confidence'] = total_confidence / len(results) if results else 0
        
        # 计算各组准确率
        for emotion_stats in stats['by_emotion'].values():
            if emotion_stats['total'] > 0:
                emotion_stats['accuracy'] = emotion_stats['correct'] / emotion_stats['total']
        
        for mod_stats in stats['by_modification'].values():
            if mod_stats['total'] > 0:
                mod_stats['accuracy'] = mod_stats['correct'] / mod_stats['total']
        
        return stats 