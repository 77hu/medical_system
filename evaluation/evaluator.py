"""
评估器 - 评估诊断准确性和生成报告
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from configs.config import EVALUATIONS_DIR, RESULTS_DIR


class Evaluator:
    """评估诊断结果并生成报告"""
    
    def __init__(self):
        # 确保结果目录存在
        os.makedirs(EVALUATIONS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def evaluate_results(self, conversation_results: List[Dict]) -> Dict:
        """评估对话结果"""
        if not conversation_results:
            return {}
        
        # 直接使用conversation_results中的diagnosis_correct字段
        # 这些字段已经通过改进的_evaluate_diagnosis方法计算过了
        correct_count = sum(1 for conv in conversation_results if conv.get('diagnosis_correct', False))
        total_count = len(conversation_results)
        
        # 计算基础指标
        accuracy = correct_count / total_count if total_count > 0 else 0
        metrics = {
            'accuracy': accuracy,
            'precision': accuracy,  # 对于二分类问题，准确率等于精确率和召回率
            'recall': accuracy,
            'f1_score': accuracy
        }
        
        # 按不同维度分组计算
        grouped_metrics = {
            'by_emotion': self._calculate_grouped_metrics(conversation_results, 'patient_emotion'),
            'by_modification': self._calculate_grouped_metrics(conversation_results, 'modification_type')
        }
        
        # 生成完整报告
        full_report = {
            'overall_metrics': metrics,
            'grouped_metrics': grouped_metrics,
            'detailed_analysis': self._analyze_errors(conversation_results),
            'summary_statistics': self._get_summary_statistics(conversation_results)
        }
        
        return full_report
    
    def _calculate_metrics(self, predictions: List[str], true_labels: List[str]) -> Dict:
        """计算基础评估指标"""
        if not predictions or not true_labels:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        # 简单的二分类：正确/错误
        binary_predictions = [1 if pred.lower() in true.lower() or true.lower() in pred.lower() 
                            else 0 for pred, true in zip(predictions, true_labels)]
        binary_true = [1] * len(true_labels)
        
        return {
            'accuracy': accuracy_score(binary_true, binary_predictions),
            'precision': precision_score(binary_true, binary_predictions, zero_division=0),
            'recall': recall_score(binary_true, binary_predictions, zero_division=0),
            'f1_score': f1_score(binary_true, binary_predictions, zero_division=0)
        }
    
    def _calculate_grouped_metrics(self, results: List[Dict], group_key: str) -> Dict:
        """按指定键分组计算指标"""
        grouped = {}
        
        for conv in results:
            group_value = conv.get(group_key, 'unknown')
            if group_value not in grouped:
                grouped[group_value] = {'correct': 0, 'total': 0}
            
            grouped[group_value]['total'] += 1
            if conv.get('diagnosis_correct', False):
                grouped[group_value]['correct'] += 1
        
        # 计算每组的指标
        group_metrics = {}
        for group_value, data in grouped.items():
            accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
            group_metrics[group_value] = {
                'accuracy': accuracy,
                'precision': accuracy,
                'recall': accuracy,
                'f1_score': accuracy,
                'count': data['total']
            }
        
        return group_metrics
    
    def _analyze_errors(self, results: List[Dict]) -> Dict:
        """分析错误案例"""
        errors = []
        
        for conv in results:
            if not conv.get('diagnosis_correct', True):
                error_info = {
                    'case_id': conv.get('case_id'),
                    'emotion': conv.get('patient_emotion'),
                    'modification': conv.get('modification_type'),
                    'predicted': conv.get('doctor_diagnosis', {}).get('primary_diagnosis'),
                    'actual': conv.get('true_diagnosis'),
                    'confidence': conv.get('doctor_diagnosis', {}).get('confidence', 0),
                    'turns': conv.get('total_turns', 0)
                }
                errors.append(error_info)
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(results) if results else 0,
            'error_cases': errors[:10]  # 只返回前10个错误案例
        }
    
    def _get_summary_statistics(self, results: List[Dict]) -> Dict:
        """获取汇总统计信息"""
        turns = [conv.get('total_turns', 0) for conv in results]
        confidences = [conv.get('doctor_diagnosis', {}).get('confidence', 0) for conv in results]
        
        return {
            'total_cases': len(results),
            'average_turns': np.mean(turns) if turns else 0,
            'std_turns': np.std(turns) if turns else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0
        }
    
    def generate_report(self, evaluation_results: Dict, save_path: str = None):
        """生成评估报告"""
        if save_path is None:
            save_path = os.path.join(
                EVALUATIONS_DIR, 
                f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估报告已保存到: {save_path}")
        return save_path
    
    def create_visualizations(self, evaluation_results: Dict):
        """创建可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 按情绪类型的准确率对比
        self._plot_accuracy_by_emotion(evaluation_results)
        
        # 2. 按修改类型的准确率对比
        self._plot_accuracy_by_modification(evaluation_results)
        
        # 3. 整体指标雷达图
        self._plot_overall_metrics_radar(evaluation_results)
        
        # 4. 生成统计表格
        self._create_summary_table(evaluation_results)
    
    def _plot_accuracy_by_emotion(self, results: Dict):
        """绘制按情绪类型的准确率对比图"""
        emotion_metrics = results.get('grouped_metrics', {}).get('by_emotion', {})
        
        if not emotion_metrics:
            return
        
        emotions = list(emotion_metrics.keys())
        accuracies = [emotion_metrics[e].get('accuracy', 0) for e in emotions]
        counts = [emotion_metrics[e].get('count', 0) for e in emotions]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(emotions, accuracies)
        
        # 在条形图上添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}\n(n={count})',
                   ha='center', va='bottom')
        
        ax.set_xlabel('病人情绪类型')
        ax.set_ylabel('诊断准确率')
        ax.set_title('不同情绪状态下的诊断准确率对比')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'accuracy_by_emotion.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {save_path}")
    
    def _plot_accuracy_by_modification(self, results: Dict):
        """绘制按修改类型的准确率对比图"""
        mod_metrics = results.get('grouped_metrics', {}).get('by_modification', {})
        
        if not mod_metrics:
            return
        
        modifications = list(mod_metrics.keys())
        accuracies = [mod_metrics[m].get('accuracy', 0) for m in modifications]
        
        # 转换标签为中文
        label_map = {
            'original': '原始数据',
            'fake_symptoms_added': '添加假症状',
            'symptoms_removed': '删除部分症状'
        }
        modifications_cn = [label_map.get(m, m) for m in modifications]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(modifications_cn, accuracies, color=['green', 'orange', 'red'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom')
        
        ax.set_xlabel('数据修改类型')
        ax.set_ylabel('诊断准确率')
        ax.set_title('不同数据修改类型下的诊断准确率对比')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'accuracy_by_modification.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {save_path}")
    
    def _plot_overall_metrics_radar(self, results: Dict):
        """绘制整体指标雷达图"""
        overall_metrics = results.get('overall_metrics', {})
        
        if not overall_metrics:
            return
        
        # 准备数据
        categories = ['准确率', '精确率', '召回率', 'F1分数']
        values = [
            overall_metrics.get('accuracy', 0),
            overall_metrics.get('precision', 0),
            overall_metrics.get('recall', 0),
            overall_metrics.get('f1_score', 0)
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('整体诊断性能指标', size=16, y=1.1)
        
        # 添加数值标签
        for angle, value, cat in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='center')
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'overall_metrics_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {save_path}")
    
    def _create_summary_table(self, results: Dict):
        """创建汇总统计表格"""
        # 准备数据
        summary_data = []
        
        # 按情绪分组的数据
        emotion_metrics = results.get('grouped_metrics', {}).get('by_emotion', {})
        for emotion, metrics in emotion_metrics.items():
            row = {
                '分组类型': '情绪状态',
                '分组值': emotion,
                '样本数': metrics.get('count', 0),
                '准确率': f"{metrics.get('accuracy', 0):.2%}",
                'F1分数': f"{metrics.get('f1_score', 0):.2f}"
            }
            summary_data.append(row)
        
        # 按修改类型分组的数据
        mod_metrics = results.get('grouped_metrics', {}).get('by_modification', {})
        label_map = {
            'original': '原始数据',
            'fake_symptoms_added': '添加假症状',
            'symptoms_removed': '删除部分症状'
        }
        for mod, metrics in mod_metrics.items():
            row = {
                '分组类型': '数据修改',
                '分组值': label_map.get(mod, mod),
                '样本数': metrics.get('count', 0),
                '准确率': f"{metrics.get('accuracy', 0):.2%}",
                'F1分数': f"{metrics.get('f1_score', 0):.2f}"
            }
            summary_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(summary_data)
        save_path = os.path.join(RESULTS_DIR, 'summary_statistics.csv')
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"统计表格已保存: {save_path}")
        
        # 打印表格
        print("\n=== 汇总统计表 ===")
        print(df.to_string(index=False)) 