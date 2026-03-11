# 南宁电信本地化知识蒸馏与边缘部署

## 项目成果
- ✅ 718条 CoT 思维链训练数据（153个南宁排障场景种子）
- ✅ Qwen2.5-1.5B QLoRA 微调完成，<think>标签蒸馏成功
- ✅ GGUF Q4_K_M 量化，GTX 1050Ti 4GB 本地部署通过
- ✅ Ollama 本地运行，无需联网，无需API费用

## 技术栈
| 环节 | 技术 |
|------|------|
| 数据合成 | GPT-4o CoT蒸馏 |
| 数据清洗去重 | Python + sentence-transformers |
| 模型微调 | Unsloth + QLoRA + TRL（Colab T4） |
| 量化部署 | GGUF Q4_K_M + Ollama |
