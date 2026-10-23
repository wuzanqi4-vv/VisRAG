# Visual RAG baseline generation

本模块只检索、生成和留存结果，不计算准确度，不调用 LLM judge。
默认 Top-5，支持 NationStandardBench 与 DocBench、VisRAG-Ret 与 ColPali。
按照仓库 AGENTS.md，运行和测试均在服务器执行。

## 配置

服务器仓库根目录 `.env`：

```dotenv
CHAT_API_KEY=填写真实密钥
CHAT_BASE_URL=填写提供商的API根地址
CHAT_MODEL=填写准确模型标识
CHAT_API_FORMAT=chat_completions
BENCHMARK_DATA_DIR=/home/wuzanqi/Experience
BENCHMARK_MODEL_DIR=/home/wuzanqi/ModelScope
```

Base URL 包含提供商要求的版本/兼容路径，不追加 `/chat/completions`。
已有 DASHSCOPE/OPENAI/QWEN 别名仍可读取，CHAT 字段优先；进程环境优先于文件。
不会默认换成 qwen-vl-max。真实密钥不进入结果或 Git。
可用 `--env-file` 显式指定环境文件。

## 环境与命令

VisRAG 使用现有 `VisRAG` 环境。ColPali HF 使用服务器现有的 `colpali` 环境，
其中必须能够导入 `transformers.ColPaliForRetrieval` 和 `ColPaliProcessor`。
不要在原 VisRAG 环境中升级 Transformers。两个环境分别执行各自检索器。

```bash
cd /home/wuzanqi/VisRAG
conda activate VisRAG
python visrag_scripts/baseline_eval/run_eval.py --dataset nation_standard_bench --retriever visrag --top_k 5 --limit 2 --resume
```

正式实验移除 `--limit 2`，另用 `--dataset doc_bench` 运行 DocBench。
ColPali 在 `conda activate colpali` 后使用 `--retriever colpali`。
这样得到两个数据集乘两个检索器的四组结果。
若提供商要求显式关闭 Qwen 思考模式，添加 `--thinking-mode disabled`；
默认不发送此提供商专用参数，保持 `provider_default`，并将选项计入运行指纹。
`--max-tokens` 默认 1024，`--timeout` 默认 120 秒，`--max-retries` 默认 3 次。
同一对比实验保持这些配置相同。

## 续跑与输出

输出默认在 `artifacts/baseline_eval/`，可用 `BENCHMARK_ARTIFACT_DIR` 指定位置。
页面缓存由 PDF 内容及 DPI 定位；向量缓存按检索器、模型和编码代码隔离。
模型标识包含配置内容及权重文件大小/修改时间，不宣称已经校验整份权重 SHA。
每次只在内存中打开少量页图，ColPali 向量保存在 CPU 上逐页评分。

运行目录：`runs/<dataset>/<retriever>/top5_<fingerprint>/`。
数据内容、模型、代码、提示词、API 模型/地址/格式与输出设置变化会产生新运行。
小样本和全量结果隔离。`--no-resume` 显式创建新的运行实例。
相同配置默认跳过已成功或截断的题，保留失败及原始响应。
失败任务在剩余尝试预算内恢复；预算耗尽或永久失败后，使用 `--retry-failed`
显式授予新一轮有上限的尝试。401/403 暂停本次执行，剩余题目保留待处理。
同一个实验目录只允许一个进程生成。

- `state.sqlite3`：权威状态，事务保存联合问题键、检索、每次请求和最终结果。
- `predictions.jsonl`：全部问题、参考答案/元数据、最终回答、原始响应和结束原因。
- `evaluation_input.jsonl`：供后续统一 LLM 评测；失败/待处理题也保留，不改变全集。
- `api_requests.jsonl`：每次 Qwen 请求与重试的实际 usage、时间和状态。
- `retrieved_pages.jsonl`：真实物理页码、排名分数与检索耗时。
- `results.csv`：便于人工查看的逐题结果。
- `summary.json`：从数据库重建的全运行已知 token、完成数量、失败与耗时汇总。
- `sessions/`：各次执行的阶段时间，不用空续跑覆盖此前耗时。
- `manifest.json`：不含密钥的运行配置和输入指纹。

Qwen usage 缺失时 token 为 null，不估算，不伪装成 0。
只汇总 API 实际返回的已知值，并报告缺失次数；本地检索器没有 API token。
截断回答标为 truncated，正常拒答保留原文，失败回答为 null。
原始参考答案只用于导出，不进入生成请求。

无需密钥或模型即可重新导出：

```bash
python visrag_scripts/baseline_eval/run_eval.py --export-only --run-dir /实际运行目录
```

旧版 `visrag_scripts/baseline_eval/results` 和 `cache` 保留原样。
由于旧记录缺少可靠的问题键和实验指纹，不自动认领为新实验的已完成状态，
避免错跳题和混合模型；新目录会重新生成，正式发起前应知悉这点。
程序遇到服务端已处理但响应未落盘的中断窗口，仍可能重复调用；
未知调用用量不能补算。不要同时同步服务器活动 SQLite 数据库。

## 服务器离线验证

```bash
cd /home/wuzanqi/VisRAG/visrag_scripts/baseline_eval
/home/wuzanqi/miniconda3/envs/VisRAG/bin/python -m unittest test_regressions -v
/home/wuzanqi/miniconda3/envs/VisRAG/bin/python test_sanity.py
```

这些检查不会调用 API。`test_api.py` 仅在显式执行时发出一次真实请求，
普通导入和 unittest/pytest 收集不会触发付费调用。