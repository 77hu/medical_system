import os
import sys
import json
from datetime import datetime

# 1. 获取当前脚本（run_chatbot.py）的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取脚本所在目录（scripts/）
current_script_dir = os.path.dirname(current_script_path)
# 3. 获取src/目录（scripts/的父目录）
src_dir = os.path.dirname(current_script_dir)
# 4. 获取项目根目录（src/的父目录，可选：也可直接添加src_dir）
project_root = os.path.dirname(src_dir)
# 5. 将src/目录加入sys.path（核心：让Python能找到src下的所有包）
sys.path.append(src_dir)

# ========== 改为绝对导入（从src/下的包开始） ==========
from data_processing.data_loader import DataLoader
from data_processing.symptom_generator import SymptomGenerator
from conversation_manager import ConversationManager
from evaluation.evaluator import Evaluator
from utils.performance_tracker import PerformanceTracker
from configs.config import PATIENT_EMOTIONS, RESULTS_DIR,FAKE_DATA_PATHS


def read_fake_data_evaluate():
    main_path=os.path.dirname(os.path.abspath(__file__)).split('step1')[0]
    save_path = os.path.join(main_path,FAKE_DATA_PATHS)
    #读取伪数据进行评估
    all_results = []
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            modified_datasets = json.load(f)
        # print(modified_datasets)
        print("\n--- 测试添加假症状的数据 ---")
        conv_manager = ConversationManager()
        results_fake = conv_manager.batch_consultations(
            cases=modified_datasets['with_fake_symptoms'][:3],
            emotions=["calm", "anxious"]  # 减少组合数
        )
        all_results.extend(results_fake)
        # 4. 评估结果
        print(f"\n{'📊 4. 评估结果':=^60}")
        evaluator = Evaluator()
        # 计算评估指标
        evaluation_results = evaluator.evaluate_results(all_results)
        print(evaluation_results)
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
        
        
    except Exception as e:
        print(f"读取失败：{str(e)}")
        return None


if __name__=="__main__":
    read_fake_data_evaluate()
    