"""
性能跟踪器 - 用于保存和分析性能数据
"""
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.performance_dir = os.path.join(results_dir, "performance")
        os.makedirs(self.performance_dir, exist_ok=True)
    
    def save_performance_summary(self, conversation_results: List[Dict]) -> str:
        """保存性能汇总报告"""
        if not conversation_results:
            return ""
        
        # 收集所有性能数据
        performance_data = []
        total_stats = {
            'total_tokens': 0,
            'total_time': 0,
            'total_calls': 0,
            'total_conversations': len(conversation_results)
        }
        
        for conv in conversation_results:
            if 'performance_stats' in conv:
                doctor_stats = conv['performance_stats']['doctor']
                patient_stats = conv['performance_stats']['patient']
                
                conv_tokens = doctor_stats['total_tokens'] + patient_stats['total_tokens']
                conv_time = doctor_stats['total_time'] + patient_stats['total_time']
                conv_calls = doctor_stats['api_calls'] + patient_stats['api_calls']
                
                performance_data.append({
                    'case_id': conv['case_id'],
                    'emotion': conv['patient_emotion'],
                    'modification_type': conv.get('modification_type', 'original'),
                    'diagnosis_correct': conv.get('diagnosis_correct', False),
                    'total_tokens': conv_tokens,
                    'total_time': conv_time,
                    'total_calls': conv_calls,
                    'tokens_per_second': conv_tokens / max(conv_time, 0.001),
                    'doctor_tokens': doctor_stats['total_tokens'],
                    'patient_tokens': patient_stats['total_tokens'],
                    'doctor_time': doctor_stats['total_time'],
                    'patient_time': patient_stats['total_time']
                })
                
                total_stats['total_tokens'] += conv_tokens
                total_stats['total_time'] += conv_time
                total_stats['total_calls'] += conv_calls
        
        # 计算平均值
        avg_stats = {
            'avg_tokens_per_conversation': total_stats['total_tokens'] / max(total_stats['total_conversations'], 1),
            'avg_time_per_conversation': total_stats['total_time'] / max(total_stats['total_conversations'], 1),
            'avg_calls_per_conversation': total_stats['total_calls'] / max(total_stats['total_conversations'], 1),
            'overall_tokens_per_second': total_stats['total_tokens'] / max(total_stats['total_time'], 0.001)
        }
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {**total_stats, **avg_stats},
            'detailed_data': performance_data
        }
        
        # 保存JSON报告
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = os.path.join(self.performance_dir, filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存CSV数据
        if performance_data:
            csv_filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(self.performance_dir, csv_filename)
            df = pd.DataFrame(performance_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return json_path
    
    def print_performance_summary(self, conversation_results: List[Dict]):
        """打印性能汇总信息"""
        if not conversation_results:
            return
        
        print(f"\n{'📊 性能统计汇总':=^80}")
        
        # 收集数据
        total_tokens = 0
        total_time = 0
        total_calls = 0
        correct_count = 0
        
        for conv in conversation_results:
            if 'performance_stats' in conv:
                doctor_stats = conv['performance_stats']['doctor']
                patient_stats = conv['performance_stats']['patient']
                
                total_tokens += doctor_stats['total_tokens'] + patient_stats['total_tokens']
                total_time += doctor_stats['total_time'] + patient_stats['total_time']
                total_calls += doctor_stats['api_calls'] + patient_stats['api_calls']
                
            if conv.get('diagnosis_correct', False):
                correct_count += 1
        
        # 打印汇总
        print(f"🔢 总Token消耗: {total_tokens:,}")
        print(f"⏱️  总耗时: {total_time:.2f}s")
        print(f"📞 总API调用: {total_calls} 次")
        print(f"🎯 诊断准确率: {correct_count}/{len(conversation_results)} ({correct_count/len(conversation_results):.1%})")
        print(f"🚀 平均处理速度: {total_tokens/max(total_time, 0.001):.1f} tokens/s")
        print(f"💰 平均每次对话: {total_tokens/len(conversation_results):.0f} tokens")
        
        # 按情绪分组统计
        emotion_stats = {}
        for conv in conversation_results:
            emotion = conv.get('patient_emotion', 'unknown')
            if emotion not in emotion_stats:
                emotion_stats[emotion] = {'count': 0, 'tokens': 0, 'time': 0, 'correct': 0}
            
            emotion_stats[emotion]['count'] += 1
            if conv.get('diagnosis_correct', False):
                emotion_stats[emotion]['correct'] += 1
                
            if 'performance_stats' in conv:
                doctor_stats = conv['performance_stats']['doctor']
                patient_stats = conv['performance_stats']['patient']
                emotion_stats[emotion]['tokens'] += doctor_stats['total_tokens'] + patient_stats['total_tokens']
                emotion_stats[emotion]['time'] += doctor_stats['total_time'] + patient_stats['total_time']
        
        print(f"\n📈 按情绪类型统计:")
        for emotion, stats in emotion_stats.items():
            accuracy = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
            avg_tokens = stats['tokens'] / stats['count'] if stats['count'] > 0 else 0
            print(f"   {emotion}: {accuracy:.1%} 准确率, {avg_tokens:.0f} tokens/次")
        
        print("="*80) 