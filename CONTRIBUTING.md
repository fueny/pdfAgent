# 贡献指南

感谢您对 PDF 分析系统的关注！我们欢迎各种形式的贡献，包括但不限于：

- 报告 Bug
- 提交功能请求
- 提交代码改进
- 改进文档
- 分享使用案例

## 开发环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/pdf-analysis-system.git
cd pdf-analysis-system
```

2. 创建虚拟环境：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖项：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，设置你的 API 密钥和其他配置
```

## 代码风格

- 请遵循 PEP 8 代码风格指南
- 使用有意义的变量名和函数名
- 为函数和类添加文档字符串
- 使用中文注释，使代码更易于理解

## 提交 Pull Request

1. 创建一个新分支：
```bash
git checkout -b feature/your-feature-name
```

2. 进行更改并提交：
```bash
git add .
git commit -m "添加新功能：功能描述"
```

3. 推送到你的分支：
```bash
git push origin feature/your-feature-name
```

4. 在 GitHub 上创建一个 Pull Request

## 注意事项

- 不要提交 `.env` 文件，它包含敏感信息
- 不要提交大型二进制文件或测试 PDF 文件
- 确保你的代码通过所有测试
- 如果添加新功能，请同时更新文档

## 报告 Bug

如果你发现了 Bug，请创建一个 Issue，并包含以下信息：

- Bug 描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（操作系统、Python 版本等）
- 如果可能，提供截图或日志

## 联系方式

如果你有任何问题，可以通过以下方式联系我们：

- 创建 GitHub Issue
- 发送邮件至：your-email@example.com

感谢您的贡献！
