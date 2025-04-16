"""
PDF 分析系统

这是 PDF 分析系统的主模块，用于协调分析 PDF 文档、生成摘要和提取见解的工作流程。
支持 OpenAI GPT 和 xAI 的 Grok-3 API，可根据配置自动切换。

用法：
    python main.py <pdf文件路径> [<pdf文件路径> ...]
"""

import os
import sys
import warnings
from typing import Dict, List, Any, Optional, TypedDict
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END

# 抑制 PyPDF2 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

# Import configuration
import config

# 定义状态类型
class PDFAnalysisState(TypedDict):
    file_paths: List[str]                                      # PDF文件路径列表
    status: NotRequired[str]                                  # 处理状态
    message: NotRequired[str]                                 # 状态消息
    extracted_texts: NotRequired[Dict[str, str]]              # 提取的文本内容
    text_chunks: NotRequired[List[Dict[str, Any]]]            # 分块后的文本
    vector_db: NotRequired[Any]                               # 向量数据库
    summaries: NotRequired[Dict[str, str]]                    # 生成的摘要
    golden_sentences: NotRequired[Dict[str, List[Dict[str, str]]]]  # 黄金句子
    similarity_analysis: NotRequired[Any]                     # 相似度分析结果
    memory_manager: NotRequired[Any]                          # 内存管理器

# 导入所有系统组件
from pdf_processing import PDFProcessor, pdf_processor_node         # PDF处理器和工作流节点
from pdf_processing import PDFTextSplitter, text_splitter_node      # 文本分割器和工作流节点
from vector_storage import VectorDatabaseManager, vector_db_node    # 向量数据库管理器和工作流节点
from content_analysis import ContentAnalyzer, content_analyzer_node  # 内容分析器和工作流节点
from content_analysis import MemoryManager, memory_manager_node, initialize_memory_node, save_results_node  # 内存管理相关组件


class PDFAnalysisSystem:
    """
    PDF 分析系统的主类

    该类协调分析 PDF 文档、生成摘要和提取见解的工作流程。
    它使用 LangGraph 构建一个工作流图，将各个处理节点连接起来。
    """

    def __init__(self):
        """
        初始化 PDF 分析系统
        """
        # 创建工作流图
        self.workflow = self._create_workflow()  # 调用私有方法创建工作流图

    def _create_workflow(self) -> StateGraph:
        """
        创建 PDF 分析的工作流图

        这个方法使用 LangGraph 创建一个状态图，定义了各个处理节点之间的连接和流程。

        返回：
            StateGraph: 编译后的工作流图对象
        """
        # 定义状态图的模式
        workflow = StateGraph(PDFAnalysisState)  # 使用我们定义的状态类型

        # 将处理节点添加到图中
        workflow.add_node("initialize_memory", initialize_memory_node)  # 初始化内存节点
        workflow.add_node("process_pdfs", pdf_processor_node)         # PDF处理节点
        workflow.add_node("split_text", text_splitter_node)           # 文本分割节点
        workflow.add_node("embed_text", vector_db_node)               # 向量嵌入节点
        workflow.add_node("analyze_content", content_analyzer_node)   # 内容分析节点
        workflow.add_node("save_results", save_results_node)          # 保存结果节点

        # 定义节点之间的连接
        workflow.add_edge("initialize_memory", "process_pdfs")   # 初始化内存 -> 处理PDF
        workflow.add_edge("process_pdfs", "split_text")         # 处理PDF -> 分割文本
        workflow.add_edge("split_text", "embed_text")           # 分割文本 -> 向量嵌入
        workflow.add_edge("embed_text", "analyze_content")      # 向量嵌入 -> 内容分析
        workflow.add_edge("analyze_content", "save_results")    # 内容分析 -> 保存结果
        workflow.add_edge("save_results", END)                  # 保存结果 -> 结束

        # 定义条件连接（错误处理）
        def check_error(state):
            """检查状态中是否有错误"""
            return "error" if state.get("status") == "error" else "continue"  # 如果状态为错误返回错误路径，否则继续

        # 添加错误处理的条件连接
        workflow.add_conditional_edges(
            "process_pdfs",           # 从这个节点开始
            check_error,              # 使用错误检查函数
            {
                "error": END,         # 如果有错误，直接结束
                "continue": "split_text"  # 如果没有错误，继续到下一个节点
            }
        )

        workflow.add_conditional_edges(
            "split_text",             # 从文本分割节点
            check_error,              # 使用错误检查函数
            {
                "error": END,         # 如果有错误，直接结束
                "continue": "embed_text"  # 如果没有错误，继续到下一个节点
            }
        )

        workflow.add_conditional_edges(
            "embed_text",             # 从向量嵌入节点
            check_error,              # 使用错误检查函数
            {
                "error": END,         # 如果有错误，直接结束
                "continue": "analyze_content"  # 如果没有错误，继续到下一个节点
            }
        )

        # 设置工作流的入口点
        workflow.set_entry_point("initialize_memory")  # 从初始化内存节点开始

        # 编译工作流图，生成可执行的工作流
        return workflow.compile()  # 返回编译后的工作流

    def analyze_pdfs(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        分析 PDF 文档

        这是系统的主要入口点，接收一个或多个 PDF 文件路径，返回分析结果。
        它会调用工作流执行完整的分析过程。

        参数：
            file_paths (List[str]): PDF 文件路径列表

        返回：
            Dict[str, Any]: 分析结果，包含摘要、黄金句子和相似度分析
        """
        # 验证文件路径
        valid_paths = []
        for path in file_paths:  # 遍历所有文件路径
            if os.path.exists(path) and os.path.isfile(path):  # 检查文件是否存在且是否是文件
                valid_paths.append(path)  # 添加到有效路径列表
            else:
                print(f"警告: 文件不存在: {path}")  # 输出警告信息

        if not valid_paths:  # 如果没有有效的文件路径
            print("错误: 没有提供有效的 PDF 文件")
            return {  # 返回错误状态
                "status": "error",  # 设置状态为错误
                "message": "没有提供有效的 PDF 文件"  # 错误消息
            }

        # 创建初始状态
        initial_state = {"file_paths": valid_paths}  # 将有效的文件路径添加到初始状态中

        # 运行工作流
        try:
            result = self.workflow.invoke(initial_state)  # 调用工作流处理文件
            return result  # 返回处理结果
        except Exception as e:
            print(f"错误运行工作流: {str(e)}")
            return {  # 返回错误状态
                "status": "error",  # 设置状态为错误
                "message": f"错误运行工作流: {str(e)}"  # 错误消息
            }


def main():
    """
    PDF 分析系统的主函数

    这个函数是命令行工具的入口点，处理命令行参数并调用分析系统。
    """
    # 检查命令行参数
    if len(sys.argv) < 2:  # 如果没有提供文件路径
        print("用法: python main.py <pdf文件路径> [<pdf文件路径> ...]")
        sys.exit(1)  # 退出程序

    # 从命令行参数获取 PDF 文件路径
    file_paths = sys.argv[1:]  # 第一个参数之后的所有参数都是文件路径

    # 创建 PDF 分析系统
    system = PDFAnalysisSystem()  # 创建分析系统实例

    # 分析 PDF 文件
    print(f"正在分析 {len(file_paths)} 个 PDF 文件...")
    result = system.analyze_pdfs(file_paths)  # 调用分析方法

    # 打印结果状态
    print(f"状态: {result.get('status', 'unknown')}")
    print(f"消息: {result.get('message', '')}")

    # 打印输出文件路径
    output_path = result.get("output_path", config.OUTPUT_FILE)  # 获取输出文件路径
    if os.path.exists(output_path):  # 检查文件是否存在
        print(f"结果保存到: {output_path}")

    return result


if __name__ == "__main__":
    main()  # 如果直接运行这个文件，则调用主函数
