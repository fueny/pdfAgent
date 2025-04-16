"""
API 服务示例脚本

这个脚本演示了如何使用 PDF 分析 API 服务。
"""

import os
import sys
import warnings
import io
from contextlib import contextmanager

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 抑制 PyPDF2 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

# 创建一个上下文管理器来隐藏 stderr 输出
@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr

# 当作为模块导入时使用 api.api_service
# 当直接运行时使用相对导入
try:
    from api.api_service import PDFAnalysisAPI, LLMServiceFactory
except ModuleNotFoundError:
    from api_service import PDFAnalysisAPI, LLMServiceFactory


def main():
    """
    主函数
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python api_example.py <pdf文件路径> [<问题>]")
        sys.exit(1)

    # 获取 PDF 文件路径
    file_path = sys.argv[1]

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)

    # 创建 LLM 服务
    llm_service = LLMServiceFactory.create_service()

    # 创建 PDF 分析 API 服务
    api = PDFAnalysisAPI(llm_service)

    # 如果提供了问题，则回答问题
    if len(sys.argv) > 2:
        question = sys.argv[2]
        print(f"问题: {question}")
        print("\n回答:")

        # 使用上下文管理器隐藏 stderr 输出
        with suppress_stderr():
            answer = api.ask_question(file_path, question)
        print(answer)
    else:
        # 否则，显示摘要和黄金句子
        print(f"分析 PDF 文件: {file_path}")
        print("\n摘要:")

        # 使用上下文管理器隐藏 stderr 输出
        with suppress_stderr():
            summary = api.get_summary(file_path)
        print(summary)

        print("\n黄金句子:")
        with suppress_stderr():
            golden_sentences = api.get_golden_sentences(file_path)
        for i, sentence in enumerate(golden_sentences):
            print(f"{i+1}. {sentence['sentence']}")
            print(f"   上下文: {sentence['context']}")
            print(f"   原因: {sentence['reason']}")
            print()


if __name__ == "__main__":
    main()
