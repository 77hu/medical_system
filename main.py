"""
主程序 - AI医生与病人对话系统
"""
import os
import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.data_loader import DataLoader
from data_processing.symptom_generator import SymptomGenerator
from conversation_manager import ConversationManager
from evaluation.evaluator import Evaluator
from utils.performance_tracker import PerformanceTracker
from configs.config import PATIENT_EMOTIONS, RESULTS_DIR


def main():
    """主函数"""
    print(f"{'🏥 AI医生与病人对话系统':=^80}")
    print(f"🚀 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 医生模型: DeepSeek-R1-Distill-Qwen-32B")
    print(f"🤒 病人模型: DeepSeek-R1-Distill-Qwen-14B")
    print("="*80)
    
    # 1. 加载数据
    print(f"\n{'📂 1. 加载数据集':=^60}")
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("❌ 错误：未能加载任何数据集")
        return
    
    # 2. 生成修改后的数据集
    print(f"\n{'🔄 2. 生成修改后的数据集':=^60}")
    symptom_gen = SymptomGenerator()
    
    # 选择要处理的数据集和样本数
    selected_dataset = "nejmai"  # 可以改为其他数据集
    if selected_dataset in datasets:
        # 只取前5个案例进行测试（实际使用时可以增加）
        test_cases = datasets[selected_dataset][:5]
        print(f"✅ 选择了 {len(test_cases)} 个案例进行测试")
        
        # 生成修改后的数据
        modified_datasets = symptom_gen.generate_modified_datasets(test_cases)
        print(f"📊 数据集统计:")
        print(f"   📝 原始数据: {len(modified_datasets['original'])} 个案例")
        print(f"   ➕ 添加假症状: {len(modified_datasets['with_fake_symptoms'])} 个案例")
        print(f"   ➖ 删除部分症状: {len(modified_datasets['with_missing_symptoms'])} 个案例")
    else:
        print(f"未找到数据集: {selected_dataset}")
        return
    
    # 3. 进行对话实验
    print("\n3. 开始对话实验...")
    conv_manager = ConversationManager()
    
    # 选择要测试的情绪类型
    test_emotions = ["calm", "anxious", "distrustful"]  # 可以添加更多
    
    all_results = []
    
    # 测试原始数据
    print("\n--- 测试原始数据 ---")
    results_original = conv_manager.batch_consultations(
        cases=modified_datasets['original'][:3],  # 只测试前3个
        emotions=test_emotions
    )
    all_results.extend(results_original)
    
    # 测试添加假症状的数据
    print("\n--- 测试添加假症状的数据 ---")
    results_fake = conv_manager.batch_consultations(
        cases=modified_datasets['with_fake_symptoms'][:3],
        emotions=["calm", "anxious"]  # 减少组合数
    )
    all_results.extend(results_fake)
    
    # 测试删除症状的数据
    print("\n--- 测试删除症状的数据 ---")
    results_missing = conv_manager.batch_consultations(
        cases=modified_datasets['with_missing_symptoms'][:3],
        emotions=["calm", "anxious"]
    )
    all_results.extend(results_missing)
    
    # 4. 评估结果
    print(f"\n{'📊 4. 评估结果':=^60}")
    evaluator = Evaluator()
    
    # 计算评估指标
    evaluation_results = evaluator.evaluate_results(all_results)
    
    # 生成报告
    report_path = evaluator.generate_report(evaluation_results)
    print(f"📋 评估报告已保存: {report_path}")
    
    # 创建可视化
    print(f"\n{'📈 5. 创建可视化图表':=^60}")
    evaluator.create_visualizations(evaluation_results)
    
    # 6. 性能统计
    print(f"\n{'⚡ 6. 性能统计':=^60}")
    performance_tracker = PerformanceTracker(RESULTS_DIR)
    performance_path = performance_tracker.save_performance_summary(all_results)
    performance_tracker.print_performance_summary(all_results)
    print(f"💾 性能报告已保存: {performance_path}")
    
    # 7. 实验总结
    print(f"\n{'🎯 实验总结':=^80}")
    print(f"📊 总对话数: {len(all_results)}")
    print(f"🎯 整体准确率: {evaluation_results['overall_metrics']['accuracy']:.2%}")
    
    print(f"\n📈 按情绪类型的准确率:")
    for emotion, metrics in evaluation_results['grouped_metrics']['by_emotion'].items():
        print(f"   😊 {emotion}: {metrics['accuracy']:.2%} (n={metrics['count']})")
    
    print(f"\n🔄 按修改类型的准确率:")
    for mod_type, metrics in evaluation_results['grouped_metrics']['by_modification'].items():
        mod_emoji = {"original": "📝", "fake_symptoms_added": "➕", "symptoms_removed": "➖"}.get(mod_type, "🔹")
        print(f"   {mod_emoji} {mod_type}: {metrics['accuracy']:.2%} (n={metrics['count']})")
    
    print(f"\n💾 所有结果已保存到: {RESULTS_DIR}")
    print(f"🏁 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def run_single_case_demo():
    """运行单个案例的演示"""
    print("=== 单个案例演示 ===")
    
    # 创建一个简单的测试案例
    test_case = {
        'case_id': 'demo_001',
        'chief_complaint': '持续性头痛3天',
        'symptoms': ['头痛', '恶心', '畏光', '颈部僵硬'],
        'history': '既往体健，无高血压、糖尿病史',
        'diagnosis': '偏头痛'
    }
    
    # 创建对话管理器
    conv_manager = ConversationManager()
    
    # 测试不同情绪状态
    print("\n1. 平和的病人:")
    result_calm = conv_manager.conduct_consultation(
        case_info=test_case,
        patient_emotion="calm"
    )
    
    print("\n\n2. 焦虑的病人:")
    result_anxious = conv_manager.conduct_consultation(
        case_info=test_case,
        patient_emotion="anxious"
    )
    
    # 添加假症状测试
    print("\n\n3. 添加假症状的病人:")
    symptom_gen = SymptomGenerator()
    fake_symptoms = symptom_gen.generate_fake_symptoms(test_case)
    test_case_fake = test_case.copy()
    test_case_fake['fake_symptoms'] = fake_symptoms
    test_case_fake['modification_type'] = 'fake_symptoms_added'
    
    result_fake = conv_manager.conduct_consultation(
        case_info=test_case_fake,
        patient_emotion="calm"
    )
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    # 运行主程序
    # main()
    
    # 或者运行单个案例演示
    run_single_case_demo() 