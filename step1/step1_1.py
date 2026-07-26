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

def read_fake_data():
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
        # print(modified_datasets)
        main_path=os.path.dirname(os.path.abspath(__file__)).split('step1')[0]
        save_path = os.path.join(main_path,FAKE_DATA_PATHS)
        # print(save_path)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(
                     modified_datasets,          # 要保存的数据
                    f,             # 文件句柄
                    ensure_ascii=False,  # 不转义ASCII（保留原始字符，比如英文/特殊符号）
                    indent=4,      # 缩进4个空格，JSON文件更易读
                    sort_keys=False  # 不排序字典的key，保持原始顺序
                )
            print(f"JSON文件已成功保存到：{save_path}")
        except Exception as e:
            print(f"保存失败：{str(e)}")
    else:
        print(f"未找到数据集: {selected_dataset}")
        return


if __name__ == "__main__":
    read_fake_data()