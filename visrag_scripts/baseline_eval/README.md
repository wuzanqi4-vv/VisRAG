# Visual RAG Baseline Evaluation

本模块提供针对视觉多模态文档 RAG 的对比评测脚本，严格遵循扁平化、模块化设计（类似 `visrag_scripts/demo/visrag_pipeline`），支持 **VisRAG-Ret** 和 **ColPali v1.3** 两个视觉检索基线，配合 **Qwen 3.5 / Qwen-VL** 生成模型。

## 目录结构

```text
visrag_scripts/baseline_eval/
├── config.py             # 全局路径、API配置、Top-K与超参数
├── data_loader.py        # 统一加载 NationStandardBench 与 DocBench
├── render_pdf.py         # PDF 页面 200 DPI 渲染与缓存
├── retriever_visrag.py   # VisRAG-Ret 双塔稠密向量单文档 Top-K 检索
├── retriever_colpali.py  # ColPali v1.3 多向量 late interaction (MaxSim) 检索
├── generate_qwen.py      # Qwen-VL 多图输入与问答生成 (带重试与断点记录)
├── run_eval.py           # 统一主入口命令行脚本
└── README.md             # 本说明文档
```

## 评测执行命令

### 1. NationStandardBench (国标测试集, 471 题)

- **VisRAG-Ret + Qwen**:
```bash
python visrag_scripts/baseline_eval/run_eval.py \
    --dataset nation_standard_bench \
    --retriever visrag \
    --top_k 5 \
    --resume
```

- **ColPali v1.3 + Qwen**:
```bash
python visrag_scripts/baseline_eval/run_eval.py \
    --dataset nation_standard_bench \
    --retriever colpali \
    --top_k 5 \
    --resume
```

### 2. DocBench (229 篇英文复杂文档, 1102 题)

- **VisRAG-Ret + Qwen**:
```bash
python visrag_scripts/baseline_eval/run_eval.py \
    --dataset doc_bench \
    --retriever visrag \
    --top_k 5 \
    --resume
```

- **ColPali v1.3 + Qwen**:
```bash
python visrag_scripts/baseline_eval/run_eval.py \
    --dataset doc_bench \
    --retriever colpali \
    --top_k 5 \
    --resume
```

### 3. 小规模冒烟测试 (Smoke Test)
使用 `--limit 2` 参数快速验证端到端运行流程：
```bash
python visrag_scripts/baseline_eval/run_eval.py \
    --dataset nation_standard_bench \
    --retriever visrag \
    --top_k 5 \
    --limit 2
```

## 结果与日志

执行结果会自动保存在 `visrag_scripts/baseline_eval/results/` 下：
- `predictions.jsonl`: 逐题记录生成结果、耗时、Token消耗及答案对比。
- `retrieved_pages.jsonl`: 详细记录每道题检索得到的 Top-5 页面页码及相似度得分。
- `summary.json`: 评测总体完成度与统计汇总。
