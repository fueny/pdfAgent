"""
API 服务模块

这个模块提供了与 OpenAI GPT 和 Grok API 的接口，用于处理 PDF 分析请求。
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
import time

# 导入配置
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.file_utils import ensure_directory_exists


class LLMService:
    """
    语言模型服务基类

    这个类提供了与语言模型交互的基本功能。
    """

    def __init__(self, model_name: str = None):
        """
        初始化语言模型服务

        Args:
            model_name (str, optional): 模型名称，如果为 None，则使用配置中的默认值
        """
        self.model_name = model_name

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        生成文本（需要在子类中实现）

        Args:
            prompt (str): 提示文本
            **kwargs: 其他参数

        Returns:
            str: 生成的文本
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        生成 JSON 格式的响应（需要在子类中实现）

        Args:
            prompt (str): 提示文本
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 生成的 JSON 对象
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class OpenAIService(LLMService):
    """
    OpenAI GPT 服务

    这个类提供了与 OpenAI GPT API 交互的功能。
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化 OpenAI 服务

        Args:
            model_name (str, optional): 模型名称，如果为 None，则使用配置中的默认值
            api_key (str, optional): API 密钥，如果为 None，则使用配置中的值
        """
        super().__init__(model_name or config.OPENAI_MODEL)
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = "https://api.openai.com/v1"

    def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
        """
        使用 OpenAI GPT 生成文本

        Args:
            prompt (str): 提示文本
            temperature (float, optional): 温度参数，控制随机性，默认为 0.2
            max_tokens (int, optional): 最大生成的 token 数，默认为 1000

        Returns:
            str: 生成的文本
        """
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            return f"OpenAI API 密钥未设置或无效。请在 .env 文件中设置有效的 OPENAI_API_KEY。"

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"OpenAI API 错误: {response.status_code} - {response.text}"

        except Exception as e:
            return f"调用 OpenAI API 时出错: {str(e)}"

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        使用 OpenAI GPT 生成 JSON 格式的响应

        Args:
            prompt (str): 提示文本
            temperature (float, optional): 温度参数，控制随机性，默认为 0.2
            max_tokens (int, optional): 最大生成的 token 数，默认为 1000

        Returns:
            Dict[str, Any]: 生成的 JSON 对象
        """
        # 添加 JSON 格式的指示
        json_prompt = f"{prompt}\n\n请以有效的 JSON 格式返回响应。"

        text_response = self.generate_text(json_prompt, temperature, max_tokens)

        try:
            # 尝试从响应中提取 JSON
            import re
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"error": "无法从响应中提取 JSON", "raw_response": text_response}
        except json.JSONDecodeError:
            return {"error": "JSON 解析错误", "raw_response": text_response}
        except Exception as e:
            return {"error": f"处理响应时出错: {str(e)}", "raw_response": text_response}


class GrokService(LLMService):
    """
    Grok API 服务

    这个类提供了与 Grok API 交互的功能。
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化 Grok 服务

        Args:
            model_name (str, optional): 模型名称，如果为 None，则使用配置中的默认值
            api_key (str, optional): API 密钥，如果为 None，则使用配置中的值
        """
        super().__init__(model_name or config.GROK_MODEL)
        self.api_key = api_key or config.GROK_API_KEY
        self.api_base = "https://api.x.ai/v1"  # xAI 的官方 API 端点

    def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
        """
        使用 Grok 生成文本

        Args:
            prompt (str): 提示文本
            temperature (float, optional): 温度参数，控制随机性，默认为 0.2
            max_tokens (int, optional): 最大生成的 token 数，默认为 1000

        Returns:
            str: 生成的文本
        """
        if not self.api_key:
            return f"Grok API 密钥未设置。请在 .env 文件中设置有效的 GROK_API_KEY。"

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Grok API 错误: {response.status_code} - {response.text}"

        except Exception as e:
            return f"调用 Grok API 时出错: {str(e)}"

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        使用 Grok 生成 JSON 格式的响应

        Args:
            prompt (str): 提示文本
            temperature (float, optional): 温度参数，控制随机性，默认为 0.2
            max_tokens (int, optional): 最大生成的 token 数，默认为 1000

        Returns:
            Dict[str, Any]: 生成的 JSON 对象
        """
        # 添加 JSON 格式的指示
        json_prompt = f"{prompt}\n\n请以有效的 JSON 格式返回响应。"

        text_response = self.generate_text(json_prompt, temperature, max_tokens)

        try:
            # 尝试从响应中提取 JSON
            import re
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"error": "无法从响应中提取 JSON", "raw_response": text_response}
        except json.JSONDecodeError:
            return {"error": "JSON 解析错误", "raw_response": text_response}
        except Exception as e:
            return {"error": f"处理响应时出错: {str(e)}", "raw_response": text_response}


class LLMServiceFactory:
    """
    语言模型服务工厂

    这个类用于创建不同类型的语言模型服务实例。
    """

    @staticmethod
    def create_service(service_type: str = "auto", model_name: str = None, api_key: str = None) -> LLMService:
        """
        创建语言模型服务实例

        Args:
            service_type (str, optional): 服务类型，可以是 "openai"、"grok" 或 "auto"，默认为 "auto"
            model_name (str, optional): 模型名称，如果为 None，则使用配置中的默认值
            api_key (str, optional): API 密钥，如果为 None，则使用配置中的值

        Returns:
            LLMService: 语言模型服务实例
        """
        if service_type == "auto":
            # 自动选择可用的服务
            # 优先使用 Grok API
            if config.is_grok_api_key_valid():
                print("Using Grok API with key:", config.GROK_API_KEY[:5] + "..." + config.GROK_API_KEY[-5:])
                return GrokService(model_name, api_key)
            elif config.is_openai_api_key_valid():
                return OpenAIService(model_name, api_key)
            else:
                # 如果都不可用，返回占位符服务
                print("No valid API keys found. Using placeholder service.")
                return GrokService(model_name, api_key)
        elif service_type == "openai":
            return OpenAIService(model_name, api_key)
        elif service_type == "grok":
            return GrokService(model_name, api_key)
        else:
            raise ValueError(f"不支持的服务类型: {service_type}")


# API 服务类，用于处理 PDF 分析请求
class PDFAnalysisAPI:
    """
    PDF 分析 API 服务

    这个类提供了 PDF 分析的 API 接口，可以用于与前端集成。
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        初始化 PDF 分析 API 服务

        Args:
            llm_service (LLMService, optional): 语言模型服务实例，如果为 None，则自动创建
        """
        self.llm_service = llm_service or LLMServiceFactory.create_service()

        # 导入 PDF 分析系统组件
        from main import PDFAnalysisSystem
        self.analysis_system = PDFAnalysisSystem()

    def analyze_pdf(self, file_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        分析 PDF 文件

        Args:
            file_paths (Union[str, List[str]]): PDF 文件路径或路径列表

        Returns:
            Dict[str, Any]: 分析结果
        """
        # 确保 file_paths 是列表
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # 分析 PDF
        result = self.analysis_system.analyze_pdfs(file_paths)

        return result

    def get_summary(self, file_path: str) -> str:
        """
        获取 PDF 文件的摘要

        Args:
            file_path (str): PDF 文件路径

        Returns:
            str: PDF 摘要
        """
        # 分析 PDF
        result = self.analyze_pdf(file_path)

        # 获取摘要
        file_name = os.path.basename(file_path)
        summaries = result.get("summaries", {})

        return summaries.get(file_name, "无法生成摘要")

    def get_golden_sentences(self, file_path: str) -> List[Dict[str, str]]:
        """
        获取 PDF 文件的黄金句子

        Args:
            file_path (str): PDF 文件路径

        Returns:
            List[Dict[str, str]]: 黄金句子列表
        """
        # 分析 PDF
        result = self.analyze_pdf(file_path)

        # 获取黄金句子
        file_name = os.path.basename(file_path)
        golden_sentences = result.get("golden_sentences", {})

        return golden_sentences.get(file_name, [])

    def ask_question(self, file_path: str, question: str) -> str:
        """
        针对 PDF 文件提问

        Args:
            file_path (str): PDF 文件路径
            question (str): 问题

        Returns:
            str: 回答
        """
        # 分析 PDF
        result = self.analyze_pdf(file_path)

        # 获取文件名和摘要
        file_name = os.path.basename(file_path)
        summary = result.get("summaries", {}).get(file_name, "")
        golden_sentences = result.get("golden_sentences", {}).get(file_name, [])

        # 构建提示
        prompt = f"""
        我有一个关于 PDF 文件 "{file_name}" 的问题。以下是该文件的摘要和关键句子：

        摘要：
        {summary}

        关键句子：
        """

        for i, sentence in enumerate(golden_sentences):
            prompt += f"{i+1}. {sentence['sentence']}\n"
            prompt += f"   上下文：{sentence['context']}\n"

        prompt += f"\n我的问题是：{question}\n\n请根据以上信息回答我的问题。"

        # 使用语言模型生成回答
        answer = self.llm_service.generate_text(prompt)

        return answer

    def compare_pdfs(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        比较多个 PDF 文件

        Args:
            file_paths (List[str]): PDF 文件路径列表

        Returns:
            Dict[str, Any]: 比较结果
        """
        if len(file_paths) < 2:
            return {"error": "需要至少两个 PDF 文件进行比较"}

        # 分析 PDF
        result = self.analyze_pdf(file_paths)

        # 获取相似度分析
        similarity_analysis = result.get("similarity_analysis", {})

        return similarity_analysis
