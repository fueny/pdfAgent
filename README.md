# 📚 PDF 分析系统

一个强大的 PDF 文档分析工具，可以自动提取摘要、关键句子，并进行多文档比较分析。支持 OpenAI GPT 和 xAI 的 Grok-3 API。

## ✨ 主要功能

- 📄 **PDF 文本提取**：高质量提取 PDF 文本内容
- 🧩 **智能分块**：将文本分割成语义块，保留上下文
- 🔍 **向量检索**：使用先进的嵌入模型进行相似度搜索
- 🤖 **多模型支持**：同时支持 OpenAI GPT 和 Grok-3 API
- 📊 **内容分析**：生成摘要、提取黄金句子、回答问题
- 🔗 **API 接口**：提供 RESTful API，方便与前端集成

## 🚀 快速开始

### 安装

1. 克隆仓库并安装依赖：
```bash
git clone https://github.com/yourusername/pdf-analysis-system.git
cd pdf-analysis-system
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，设置你的 API 密钥
```

> ⚠️ **注意**：Windows 用户需要安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 以支持 ChromaDB

### 使用方法

#### 1️⃣ 直接分析 PDF（使用 main.py）

```bash
python main.py <pdf文件路径> [<pdf文件路径> ...]
```

示例：
```bash
python main.py attention.pdf
# 或分析多个文件
python main.py attention.pdf deepseekr254.pdf
```

结果将保存在 `pdf_analysis_results.md` 文件中，包含：
- 📝 文档摘要
- ✨ 黄金句子（关键句子及其上下文和重要性解释）
- 🔄 多文档相似度分析（当分析多个文档时）

#### 2️⃣ 使用 API 接口

启动 API 服务器：
```bash
python run.py server
```

然后访问 http://localhost:5000 使用 Web 界面，或通过 API 接口调用。

#### 3️⃣ 命令行交互模式

查看摘要和黄金句子：
```bash
python run.py cli attention.pdf
```

提问关于 PDF 内容的问题：
```bash
python run.py cli attention.pdf --question "这篇论文的主要贡献是什么？"
```

#### 4️⃣ API 示例调用

直接使用 API 示例脚本：
```bash
python api/api_example.py attention.pdf "这篇论文的主要贡献是什么？"
```

## ⚙️ 配置选项

主要配置项（在 `.env` 文件中设置）：

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| OPENAI_API_KEY | OpenAI API 密钥 | - |
| GROK_API_KEY | Grok API 密钥 | - |
| DEFAULT_LLM_SERVICE | 默认 LLM 服务 | auto |
| MAX_PDF_SIZE_MB | 最大 PDF 大小 (MB) | 50 |

## 🧰 项目结构

```
├── main.py             # 主入口和工作流编排
├── run.py              # 启动脚本
├── api/                # API 相关模块
├── pdf_processing/     # PDF 处理相关模块
├── vector_storage/     # 向量存储相关模块
├── content_analysis/   # 内容分析相关模块
├── utils/              # 工具函数
└── static/             # 前端静态文件
```

## 📝 注意事项

- 请不要将 `.env` 文件提交到 Git 仓库，它包含敏感信息
- 系统会自动选择可用的 API，优先使用 OpenAI，如果不可用则使用 Grok-3
- 对于大型 PDF 文件，处理可能需要较长时间
