"""
PDF 分析 API 服务器

这个模块提供了一个简单的 HTTP API 服务器，用于处理 PDF 分析请求。
"""

import os
import json
import tempfile
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# 添加项目根目录到 Python 路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 当作为模块导入时使用 api.api_service
# 当直接运行时使用相对导入
try:
    from api.api_service import PDFAnalysisAPI, LLMServiceFactory
except ModuleNotFoundError:
    from api_service import PDFAnalysisAPI, LLMServiceFactory


app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 限制上传文件大小为 50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 创建 API 服务
api = PDFAnalysisAPI()


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    return jsonify({
        "status": "ok",
        "message": "服务正常运行"
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_pdf():
    """
    分析 PDF 文件

    请求体应该是一个 multipart/form-data 表单，包含一个或多个 PDF 文件
    """
    # 检查是否有文件
    if 'files' not in request.files:
        return jsonify({
            "status": "error",
            "message": "没有上传文件"
        }), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({
            "status": "error",
            "message": "没有选择文件"
        }), 400

    # 保存上传的文件
    file_paths = []
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

    if not file_paths:
        return jsonify({
            "status": "error",
            "message": "没有有效的 PDF 文件"
        }), 400

    try:
        # 分析 PDF
        result = api.analyze_pdf(file_paths)

        return jsonify({
            "status": "success",
            "message": "分析完成",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"分析过程中出错: {str(e)}"
        }), 500


@app.route('/api/summary', methods=['POST'])
def get_summary():
    """
    获取 PDF 文件的摘要

    请求体应该是一个 multipart/form-data 表单，包含一个 PDF 文件
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "没有上传文件"
        }), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({
            "status": "error",
            "message": "没有选择文件"
        }), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            "status": "error",
            "message": "只支持 PDF 文件"
        }), 400

    # 保存上传的文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 获取摘要
        summary = api.get_summary(file_path)

        return jsonify({
            "status": "success",
            "message": "获取摘要成功",
            "summary": summary
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取摘要时出错: {str(e)}"
        }), 500


@app.route('/api/golden_sentences', methods=['POST'])
def get_golden_sentences():
    """
    获取 PDF 文件的黄金句子

    请求体应该是一个 multipart/form-data 表单，包含一个 PDF 文件
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "没有上传文件"
        }), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({
            "status": "error",
            "message": "没有选择文件"
        }), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            "status": "error",
            "message": "只支持 PDF 文件"
        }), 400

    # 保存上传的文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 获取黄金句子
        golden_sentences = api.get_golden_sentences(file_path)

        return jsonify({
            "status": "success",
            "message": "获取黄金句子成功",
            "golden_sentences": golden_sentences
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取黄金句子时出错: {str(e)}"
        }), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    针对 PDF 文件提问

    请求体应该是一个 multipart/form-data 表单，包含一个 PDF 文件和一个问题
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "没有上传文件"
        }), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({
            "status": "error",
            "message": "没有选择文件"
        }), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            "status": "error",
            "message": "只支持 PDF 文件"
        }), 400

    # 检查是否有问题
    question = request.form.get('question', '')
    if not question:
        return jsonify({
            "status": "error",
            "message": "没有提供问题"
        }), 400

    # 保存上传的文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 获取回答
        answer = api.ask_question(file_path, question)

        return jsonify({
            "status": "success",
            "message": "回答成功",
            "question": question,
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"回答问题时出错: {str(e)}"
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_pdfs():
    """
    比较多个 PDF 文件

    请求体应该是一个 multipart/form-data 表单，包含多个 PDF 文件
    """
    # 检查是否有文件
    if 'files' not in request.files:
        return jsonify({
            "status": "error",
            "message": "没有上传文件"
        }), 400

    files = request.files.getlist('files')
    if len(files) < 2 or files[0].filename == '':
        return jsonify({
            "status": "error",
            "message": "需要至少两个 PDF 文件进行比较"
        }), 400

    # 保存上传的文件
    file_paths = []
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

    if len(file_paths) < 2:
        return jsonify({
            "status": "error",
            "message": "需要至少两个有效的 PDF 文件进行比较"
        }), 400

    try:
        # 比较 PDF
        result = api.compare_pdfs(file_paths)

        return jsonify({
            "status": "success",
            "message": "比较完成",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"比较过程中出错: {str(e)}"
        }), 500


# 添加根路由，返回前端页面
@app.route('/')
def index():
    """
    返回前端页面
    """
    return send_from_directory(app.static_folder, 'index.html')


# 添加静态文件路由
@app.route('/<path:path>')
def static_files(path):
    """
    返回静态文件
    """
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    # 获取端口，默认为 5000
    port = int(os.environ.get('PORT', 5000))

    print(f"启动服务器，访问 http://localhost:{port} 查看前端页面")

    # 启动服务器
    app.run(host='0.0.0.0', port=port, debug=True)
