# 版本备份与回退

本地目录 `D:/CodePractice/VisRAG` 用于编辑和 Git 管理。
遵循 AGENTS.MD，实验与集成测试在 `/home/wuzanqi/VisRAG` 对应的 SSH 服务器执行。

## 个人 GitHub 仓库

当前项目来源为 `OpenBMB/VisRAG`，自己的变更应推送到个人仓库。
优先使用 GitHub Desktop 的 Create a fork，将现有仓库关联到个人 fork，
保留原始提交历史；原项目远程引用可以保留为 upstream。
具体远程名称以 `git remote -v` 的实际输出为准，不能向原作者仓库推送实验变更。

计划与配置变更放在 `codex/visual-rag-baseline-plan` 分支。
每个独立功能或修正形成一条提交，服务器执行时记录使用的提交 SHA。
成功推送后才算完成 GitHub 备份；本地 commit 与远程 push 是两个独立状态。

## 可以提交的文件

- 代码、测试、文档、依赖清单和不含秘密的配置模板。
- `.env.example` 只放占位值，说明 Chat API Key、Base URL 与准确模型标识。
- 原始数据、模型、向量索引、运行数据库、完整实验输出及真实 `.env` 不进入 Git。
  后续若需分享实验结果，应单独选择导出文件，不把它们自动纳入代码提交。

## 服务器 `.env`

在服务器的 `/home/wuzanqi/VisRAG` 中，将 `.env.example` 复制为 `.env` 后填写：

```dotenv
CHAT_API_KEY=replace_with_your_api_key
CHAT_BASE_URL=https://your-api-provider.example/v1
CHAT_MODEL=replace_with_exact_model_id
CHAT_API_FORMAT=chat_completions
```

复制前确认服务器是否已有 `.env`；已有文件直接编辑，避免覆盖有效配置。
SFTP 同步时排除 `.env` 及环境专属文件，不能用本地空配置覆盖服务器密钥。
模板目前只定义拟实现的配置接口，不代表实验程序已可运行。

## 回退已提交的修改

查看需要回退的提交：

```bash
git log --oneline -n 10
```

使用 GitHub Desktop 的 History，选择目标提交并使用 Revert Changes in Commit；
也可在工作区已妥善保存后执行以下命令，将占位符替换为实际提交 SHA：

```bash
git revert <commit_sha>
git push
```

这会新增一条撤销提交，保留原历史。没有必要删除旧提交或强制推送。
若存在未提交工作，先单独提交或保存，再回退；出现冲突时核对文件后解决。
回退代码不等于回退服务器的实验输出，已生成答案按运行 ID 保留。
恢复旧版代码执行时使用对应配置与依赖，并保留独立运行记录。
