"""
基础Agent类 - 提供与大语言模型交互的基本功能
"""
import requests
import json
import time
from typing import List, Dict, Optional
import sys
sys.path.append('..')
# from configs.config import API_KEY, API_URL, TEMPERATURE
from configs.config import TEMPERATURE

class BaseAgent:
    """基础Agent类，提供与LLM交互的基本功能"""
    
    def __init__(self, model_name: str, role: str,headers:dict,API_URL:str):
    # def __init__(self,model_name:str,role:str):
        self.model_name = model_name
        self.role = role
        self.conversation_history = []
        
        # self.headers = {
        #     'Authorization': f'Bearer {API_KEY}',
        #     'Content-Type': 'application/json'
        # }
        #配置不同厂商
        self.headers=headers
        self.api_url=API_URL
       
        # 性能统计
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_time = 0.0
        self.api_calls = 0
    
    def call_llm(self, messages: List[Dict[str, str]], temperature: float = TEMPERATURE) -> str:
        """调用大语言模型API"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        try:
            # print(self.api_url,self.headers,)
            # response = requests.post(API_URL,, headers=self.headers, json=data)

            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # 记录性能数据
            end_time = time.time()
            call_time = end_time - start_time
            self.total_time += call_time
            self.api_calls += 1
            
            # 记录token使用情况
            if 'usage' in result:
                usage = result['usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                print(f"🤖 [{self.role}] {self.model_name}")
                print(f"   ⏱️  耗时: {call_time:.2f}s")
                print(f"   📊 Token: {prompt_tokens}→{completion_tokens} (总计:{total_tokens})")
                print(f"   💰 累计: {self.total_tokens} tokens, {self.api_calls} 次调用")
            
            return result['choices'][0]['message']['content']
        except Exception as e:
            end_time = time.time()
            call_time = end_time - start_time
            self.total_time += call_time
            print(f"❌ API调用错误 [{self.role}]: {e} (耗时: {call_time:.2f}s)")
            return ""
    
    def add_to_history(self, role: str, content: str):
        """添加对话到历史记录"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return {
            'model_name': self.model_name,
            'role': self.role,
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_time': self.total_time,
            'api_calls': self.api_calls,
            'avg_time_per_call': self.total_time / max(self.api_calls, 1),
            'tokens_per_second': self.total_tokens / max(self.total_time, 0.001)
        } 