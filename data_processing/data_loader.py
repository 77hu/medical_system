"""
数据加载器 - 加载和处理各种医疗数据集
"""
import json
import pandas as pd
from typing import List, Dict, Optional
import os
import sys
sys.path.append('..')
from configs.config import DATA_PATHS


class DataLoader:
    """数据加载器，支持多种数据格式"""
    
    def __init__(self):
        self.datasets = {}
    
    def load_rare_disease_data(self, file_path: str = None) -> List[Dict]:
        """加载罕见病数据集"""
        if file_path is None:
            file_path = DATA_PATHS["rare_disease"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取Cases数组
            cases = data.get('Cases', [])
            
            # 转换为统一格式
            processed_data = []
            for idx, case in enumerate(cases[:50]):  # 只取前50个案例
                # 提取基本信息
                case_id = f"rare_{idx}"
                disease_name = case.get('Final Name', '')
                
                # 从Initial Presentation中提取信息
                initial_presentation = case.get('Initial Presentation', '{}')
                try:
                    # 解析JSON字符串
                    presentation_data = json.loads(initial_presentation)
                    chief_complaint = presentation_data.get('Clinical Presentation', '')[:200] + "..."
                    history = presentation_data.get('Past Medical History', '')
                    symptoms = self._extract_symptoms_from_presentation(presentation_data.get('Clinical Presentation', ''))
                except:
                    # 如果解析失败，使用原始字符串
                    chief_complaint = initial_presentation[:200] + "..."
                    history = "无详细病史"
                    symptoms = []
                
                processed_item = {
                    'case_id': case_id,
                    'disease': disease_name,
                    'symptoms': symptoms,
                    'chief_complaint': chief_complaint,
                    'history': history,
                    'diagnosis': disease_name,
                    'source': 'rare_disease'
                }
                processed_data.append(processed_item)
            
            return processed_data
        except Exception as e:
            print(f"加载罕见病数据失败: {e}")
            return []
    
    def load_nejmai_data(self, file_path: str = None) -> List[Dict]:
        """加载NEJM AI数据集"""
        if file_path is None:
            file_path = DATA_PATHS["nejmai"]
        
        try:
            df = pd.read_csv(file_path)
            processed_data = []
            
            for _, row in df.iterrows():
                # 从case_vignette中提取症状信息
                case_vignette = row['case_vignette']
                
                processed_item = {
                    'case_id': row['case_id'],
                    'chief_complaint': self._extract_chief_complaint(case_vignette),
                    'symptoms': self._extract_symptoms(case_vignette),
                    'history': case_vignette,
                    'diagnosis': row['answer'],
                    'choices': [row[f'choice_{i}'] for i in range(1, 6) if f'choice_{i}' in row],
                    'source': 'nejmai'
                }
                processed_data.append(processed_item)
            
            return processed_data
        except Exception as e:
            print(f"加载NEJM AI数据失败: {e}")
            return []
    
    def load_usmle_derm_data(self, file_path: str = None) -> List[Dict]:
        """加载USMLE和皮肤科数据集"""
        if file_path is None:
            file_path = DATA_PATHS["usmle_derm"]
        
        try:
            df = pd.read_csv(file_path)
            processed_data = []
            
            for _, row in df.iterrows():
                case_vignette = row['case_vignette']
                
                processed_item = {
                    'case_id': row['case_id'],
                    'chief_complaint': self._extract_chief_complaint(case_vignette),
                    'symptoms': self._extract_symptoms(case_vignette),
                    'history': case_vignette,
                    'diagnosis': row['answer'],
                    'choices': [row[f'choice_{i}'] for i in range(1, 5) if f'choice_{i}' in row],
                    'category': row.get('category', 'unknown'),
                    'source': 'usmle_derm'
                }
                processed_data.append(processed_item)
            
            return processed_data[:100]  # 只加载前100个案例以加快处理
        except Exception as e:
            print(f"加载USMLE数据失败: {e}")
            return []
    
    def load_jmed_data(self) -> List[Dict]:
        """加载京东健康JMED数据集（需要从Hugging Face下载）"""
        try:
            from datasets import load_dataset
            
            # 加载数据集
            dataset = load_dataset("jdh-algo/JMED")
            
            processed_data = []
            for item in dataset['train']:  # 假设数据在train分割中
                processed_item = {
                    'case_id': item.get('id', ''),
                    'chief_complaint': item.get('chief_complaint', ''),
                    'symptoms': self._extract_symptoms_from_jmed(item),
                    'history': item.get('history', ''),
                    'diagnosis': item.get('diagnosis', ''),
                    'source': 'jmed'
                }
                processed_data.append(processed_item)
            
            return processed_data[:50]  # 只加载前50个案例
        except Exception as e:
            print(f"加载JMED数据失败: {e}")
            print("请确保已安装datasets库: pip install datasets")
            return []
    
    def _extract_chief_complaint(self, text: str) -> str:
        """从文本中提取主诉"""
        # 简单的提取逻辑，寻找"presented with"或类似的模式
        import re
        patterns = [
            r'presented with (.+?)[.]',
            r'complaining of (.+?)[.]',
            r'chief complaint[s]? (?:was|were) (.+?)[.]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 如果没有找到特定模式，返回前100个字符
        return text[:100] + "..."
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """从文本中提取症状列表"""
        # 常见症状关键词
        symptom_keywords = [
            'fever', 'pain', 'cough', 'fatigue', 'headache', 'nausea', 'vomiting',
            'diarrhea', 'rash', 'shortness of breath', 'chest pain', 'abdominal pain',
            'swelling', 'weakness', 'dizziness', 'weight loss', 'loss of appetite'
        ]
        
        symptoms = []
        text_lower = text.lower()
        
        for keyword in symptom_keywords:
            if keyword in text_lower:
                symptoms.append(keyword)
        
        return symptoms
    
    def _extract_symptoms_from_jmed(self, item: Dict) -> List[str]:
        """从JMED数据项中提取症状"""
        # 根据JMED的实际结构调整
        symptoms = []
        if 'symptoms' in item:
            symptoms = item['symptoms']
        elif 'clinical_presentation' in item:
            symptoms = self._extract_symptoms(item['clinical_presentation'])
        
        return symptoms
    
    def _extract_symptoms_from_presentation(self, text: str) -> List[str]:
        """从临床表现文本中提取症状"""
        # 罕见病特有的症状关键词
        rare_symptom_keywords = [
            'fever', 'headache', 'vomiting', 'seizure', 'weakness', 'paralysis',
            'swelling', 'edema', 'rash', 'lesion', 'mass', 'tumor', 'cyst',
            'dysphagia', 'dyspnea', 'tachycardia', 'bradycardia', 'hypertension',
            'hypotension', 'anemia', 'jaundice', 'hepatomegaly', 'splenomegaly',
            'lymphadenopathy', 'developmental delay', 'intellectual disability',
            'seizures', 'ataxia', 'tremor', 'dystonia', 'spasticity'
        ]
        
        symptoms = []
        text_lower = text.lower()
        
        for keyword in rare_symptom_keywords:
            if keyword in text_lower:
                symptoms.append(keyword)
        
        # 如果没有找到特定症状，提取一些通用描述
        if not symptoms:
            if 'pain' in text_lower:
                symptoms.append('pain')
            if 'difficulty' in text_lower:
                symptoms.append('difficulty')
            if 'abnormal' in text_lower:
                symptoms.append('abnormality')
        
        return symptoms[:5]  # 最多返回5个症状
    
    def load_all_datasets(self) -> Dict[str, List[Dict]]:
        """加载所有数据集"""
        datasets = {
            'rare_disease': self.load_rare_disease_data(),
            'nejmai': self.load_nejmai_data(),
            'usmle_derm': self.load_usmle_derm_data(),
            # 'jmed': self.load_jmed_data()  # 暂时注释掉，需要额外安装
        }
        
        # 过滤掉空数据集
        self.datasets = {k: v for k, v in datasets.items() if v}
        
        print(f"成功加载的数据集:")
        for name, data in self.datasets.items():
            print(f"- {name}: {len(data)} 个案例")
        
        return self.datasets 