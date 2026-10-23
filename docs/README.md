# 实验设计文档

[视觉文档 RAG 基线实施计划](visual_rag_baseline_plan.md) 是当前实施依据，覆盖：

- NationStandardBench、DocBench 两个数据集。
- VisRAG-Ret、ColPali 两个检索器，默认 Top-k 为 5。
- 共用 Qwen 多模态 Chat Completions API 生成答案，保存最终结果供后续统一 LLM 评测。
- 分阶段断点续跑、API 实测 token、实际耗时和结果归档。
- API Key、Base URL 和模型名通过服务器 `.env` 配置，参考仓库 `.env.example`。
- 按 AGENTS.MD 在服务器运行，本地开发与版本管理，遵循 Google Python Style Guide。

该文档基于本地数据扫描更新，取代原计划中的 Top-3、步骤编号脚本、
检索质量指标及尚未核实的数据格式假设。当前不计算准确度或调用评判模型。
目前交付的是计划与配置模板，实验代码尚未实现。

服务器配置示例见仓库根目录 `.env.example`。
版本备份与回退流程见 [版本管理说明](version_control.md)。
