"""
PDF 分析系统启动脚本

这个脚本提供了一个简单的命令行界面，用于启动 PDF 分析系统的不同组件。
"""

import os
import sys
import argparse
import subprocess
import warnings

# 抑制 PyPDF2 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")


def run_api_server(port=5000):
    """
    启动 API 服务器

    Args:
        port (int, optional): 服务器端口，默认为 5000
    """
    os.environ['PORT'] = str(port)
    print(f"启动 API 服务器，端口: {port}")
    subprocess.run([sys.executable, "api/api_server.py"])


def run_cli(pdf_path, question=None):
    """
    运行命令行界面

    Args:
        pdf_path (str): PDF 文件路径
        question (str, optional): 问题，如果提供则回答问题
    """
    if question:
        print(f"分析 PDF 文件并回答问题: {pdf_path}")
        subprocess.run([sys.executable, "api/api_example.py", pdf_path, question])
    else:
        print(f"分析 PDF 文件: {pdf_path}")
        subprocess.run([sys.executable, "api/api_example.py", pdf_path])


def run_analysis(pdf_path):
    """
    运行 PDF 分析

    Args:
        pdf_path (str): PDF 文件路径
    """
    print(f"分析 PDF 文件: {pdf_path}")
    subprocess.run([sys.executable, "main.py", pdf_path])


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="PDF 分析系统")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # API 服务器命令
    server_parser = subparsers.add_parser("server", help="启动 API 服务器")
    server_parser.add_argument("--port", type=int, default=5000, help="服务器端口，默认为 5000")

    # CLI 命令
    cli_parser = subparsers.add_parser("cli", help="运行命令行界面")
    cli_parser.add_argument("pdf_path", help="PDF 文件路径")
    cli_parser.add_argument("--question", "-q", help="问题，如果提供则回答问题")

    # 分析命令
    analysis_parser = subparsers.add_parser("analyze", help="运行 PDF 分析")
    analysis_parser.add_argument("pdf_path", help="PDF 文件路径")

    args = parser.parse_args()

    if args.command == "server":
        run_api_server(args.port)
    elif args.command == "cli":
        run_cli(args.pdf_path, args.question)
    elif args.command == "analyze":
        run_analysis(args.pdf_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
