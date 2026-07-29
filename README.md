# VMP-MemOS

## VMP-v5.5 dual-view challenger scan

V5.5 addresses the position bias observed in the completed V5.4 Dev run. It
does not lower the paper quality gate and does not use question types, answers,
gold sessions, or Test labels. The frozen V5.3.2 Dev candidate pool is reused.

- A deterministic weighted-RRF planner combines V5's hierarchical rank with
  its session-semantic rank and emits 10 unique sessions.
- The same planner code is used for V4.3; because V4.3 has no hierarchical
  session-semantic metadata, its documented behavior is an identity Top-10.
- The shared local vLLM sees exactly 10 anonymous candidates for either method.
  It must assess every challenger `C06..C10`; an incomplete scan fails closed
  and preserves the original Top-5.
- The V5.4 symbolic evidence-span boundary verifier remains unchanged.

Start the already-downloaded local model in terminal A:

```bash
cd /home/shenshifan/projects/VMP-MemOS

export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LLM_MODEL="/home/shenshifan/models/Qwen2.5-7B-Instruct"
export VMP_VLLM_ENABLE_TOOL_CALLING=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

After `curl -H "Authorization: Bearer local-vllm-key" \
http://127.0.0.1:8000/v1/models` succeeds, run Dev in terminal B:

```bash
cd /home/shenshifan/projects/VMP-MemOS

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=/home/shenshifan/models/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v55_experiment.sh
```

For an interrupted run, add `RERANK_RESUME=1` to the command. The log is
`outputs/longmemeval/logs/vmp_v55_dev_rerank.log`; the run is
`outputs/longmemeval/runs/lme_dev_vmp_v55_rerank_seed42`. Exit code 3 means
the experiment completed but the unchanged strict Dev quality gate failed.
Only a successful run creates
`outputs/longmemeval/gates/vmp_v55_seed42_dev_pass.json`.

After that receipt exists, Test remains a two-stage single-GPU workflow:

```bash
# Stop vLLM, then generate Test candidates with BGE-M3.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
STAGE=test_candidates EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v55_experiment.sh

# Restart the same vLLM, then run the sealed Test rerank and optional reader.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=/home/shenshifan/models/Qwen2.5-7B-Instruct \
STAGE=test_rerank RUN_QA=1 \
uv run --no-sync bash scripts/run_vmp_v55_experiment.sh
```

## VMP-v5.4 symbolic evidence-span binding

V5.3.2 的服务器 Dev 结果为 `Recall-All@5 = 0.9149`：相对 raw V5 只提升
1.06 个百分点、恢复 1 题且零退化，因此实验本身完成，但没有通过未降级的论文
quality gate。错误审计表明主要瓶颈不是候选召回：93/94 个可回答问题的正确 session
已存在于候选池；主要问题是 LLM 找到了正确证据文本，却把证据绑定到错误的候选 ID。

V5.4 只修改共享的 LLM 选择协议，不改变 V4.3/V5 的候选、模型或 gate：

1. selector 只看到匿名候选 `C01..C30`，每个 excerpt 被确定性切分为
   `Cxx:Sxx` 证据 span，不暴露真实 session ID。
2. Top-5 以外的候选必须返回所属证据 span 才能晋升。程序从 span 所有者反向推导
   candidate，而不是相信模型同时填写的 candidate label。
3. boundary 使用同样的 `SLOT:Sxx` 归属验证；跨候选 span、非法 label、解析失败或
   低置信度一律 fail closed，保留原始 Top-5。
4. 两个框架仍共享同一本地 vLLM、prompt、参数、reader 和安全策略。Test 在 Dev
   gate 通过前保持封存。

Dev 直接复用已完成的 V5.3.2 candidate run，因此无需再次加载 BGE-M3。先在终端 A
启动本地 vLLM（`VMP_LLM_MODEL` 必须与本地已下载模型或其缓存标识一致）：

```bash
cd /home/shenshifan/projects/VMP-MemOS

export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_VLLM_ENABLE_TOOL_CALLING=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

确认 `curl -H "Authorization: Bearer local-vllm-key" \
http://127.0.0.1:8000/v1/models` 正常后，在终端 B 运行：

```bash
cd /home/shenshifan/projects/VMP-MemOS

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v54_experiment.sh
```

中断后原命令增加 `RERANK_RESUME=1` 即可续跑。日志位于
`outputs/longmemeval/logs/vmp_v54_dev_rerank.log`，结果位于
`outputs/longmemeval/runs/lme_dev_vmp_v54_rerank_seed42`。只有生成
`outputs/longmemeval/gates/vmp_v54_seed42_dev_pass.json` 后，才能依次运行
`STAGE=test_candidates` 与 `STAGE=test_rerank`；`exit_code=3` 仍表示实验完成但
严格 gate 未通过。

Dev gate 通过后，先停止 vLLM，独占 GPU 运行 BGE-M3 Test 候选生成：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=test_candidates \
EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v54_experiment.sh
```

然后重新启动同一个 vLLM，再执行封存 Test rerank（如需同时生成统一 reader 的
最终答案，可设置 `RUN_QA=1`）：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=test_rerank \
RUN_QA=1 \
uv run --no-sync bash scripts/run_vmp_v54_experiment.sh
```

## VMP-v5.3.2 atomic evidence-set boundary

V5.3.2 针对 V5.3.1 Dev 实验中“协议已稳定但有效恢复不足”的问题做两项修复：

1. candidate runner 先检索 40 条 memory，再按 `source_session_id` 去重并回填为
   30 个唯一 session；rerank 在第一次 LLM 请求前执行严格深度预检。
2. boundary verifier 比较完整的 locked+B/P Top-5 证据集合。每个被选中的
   promotion 必须输出来自自身候选 excerpt 的逐字引用，程序在本地验证引用后才允许
   替换；引用不存在、格式错误或低置信度都会保留原始 Top-5。

V4.3 和 V5 仍使用相同的本地 vLLM、selector、atomic boundary prompt、生成参数及
fail-closed 策略。LLM 看不到框架名、question type、gold answer 或 gold session ID。
严格 Dev gate 未降低，Test 仍保持封存。

服务器必须分阶段运行，避免 BGE-M3 与 vLLM 同时占用单卡显存。第一阶段先停止
vLLM，生成新的 Dev candidate pool：

```bash
cd /home/shenshifan/projects/VMP-MemOS

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=dev_candidates \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v532_experiment.sh
```

第二阶段启动统一的本地 vLLM。在终端 A：

```bash
cd /home/shenshifan/projects/VMP-MemOS

export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_VLLM_ENABLE_TOOL_CALLING=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

确认 `curl http://127.0.0.1:8000/v1/models` 正常后，在终端 B：

```bash
cd /home/shenshifan/projects/VMP-MemOS

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v532_experiment.sh
```

Dev 日志位于 `outputs/longmemeval/logs/vmp_v532_dev_candidates.log` 和
`outputs/longmemeval/logs/vmp_v532_dev_rerank.log`。只有生成
`outputs/longmemeval/gates/vmp_v532_seed42_dev_pass.json` 后，才能继续运行
`STAGE=test_candidates` 与 `STAGE=test_rerank`。`exit_code=3` 表示实验已完成但
质量门未通过，不表示程序崩溃。

## VMP-v5.3.1 symbolic boundary replay

V5.3.1 修复了 V5.3 真实运行中暴露出的输出协议歧义。受保护的 Top-3
只以 `LOCKED-1..3` 展示，开放槽位和 promotion 分别使用
`B1/B2/P1/P2`；LLM 不再看到或返回真实 session ID。程序只接受两个合法
slot label，并在服务端映射回 session ID。非法标签、格式错误和低置信度
promotion 仍然安全回退到原始 Top-5。

Dev 阶段可以精确复用已完成 V5.3 中保存的第一阶段 selector 响应。复用由
candidate manifest SHA-256、`(source_method, question_id)` 样本身份和候选 session
集合共同校验；原 selector prompt SHA-256 保留作审计，但不要求当前代码重新序列化
出逐字节相同的 prompt。任何样本、问题或候选集合变化都会立即停止，不会静默发起
新的 selector 请求。因此 Dev replay 不使用 BGE-M3，也不会重跑第一阶段的 200 次
LLM 调用，只实时执行 boundary verifier。

服务器终端 A 启动与原实验相同的本地 vLLM：

```bash
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_VLLM_ENABLE_TOOL_CALLING=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

终端 B 复用已经存在的
`lme_dev_vmp_v53_candidates_seed42` 和 `lme_dev_vmp_v53_rerank_seed42`：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_replay \
uv run --no-sync bash scripts/run_vmp_v531_experiment.sh
```

日志位于 `outputs/longmemeval/logs/vmp_v531_dev_replay.log`，新结果位于
`outputs/longmemeval/runs/lme_dev_vmp_v531_replay_seed42`。只有生成
`outputs/longmemeval/gates/vmp_v531_seed42_dev_pass.json` 后才能运行
`STAGE=test_candidates` 和 `STAGE=test_rerank`。`exit_code=3` 明确表示实验已经
完成但严格 Dev 质量门禁未通过，不表示进程崩溃。

## VMP-v5.3 selective boundary verification

V5.3 保留框架无关的 V5.2 Top-30 selector，但在受保护的 Top-3 后开放
两个 Top-5 槽位。第二次调用同一个本地 vLLM，只查看受保护证据、原始第
4/5 名和最多两个待晋升候选，并保守选择两个 boundary sessions。非法、
格式错误或低置信度替换都会回退到原始 Top-5。

V4.3 和 V5 使用完全相同的 selector、boundary prompt、生成参数、解析器
和安全策略。两个阶段都看不到框架名、question type、gold answer 或 gold
session ID。严格 Dev gate 要求 Recall-All@5 至少 93%、相对 raw V5
至少提升 2.5 个百分点、至少恢复三题且零退化，才允许运行 Test。

服务器严格分阶段运行：

```bash
# 先停止 vLLM，使用 BGE-M3 生成 Dev candidates。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
STAGE=dev_candidates EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v53_experiment.sh

# 另一个终端启动共享本地 vLLM 后运行：
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v53_experiment.sh
```

只有生成
`outputs/longmemeval/gates/vmp_v53_seed42_dev_pass.json` 后才能执行两阶段
Test。配置与公平性约束见 `configs/vmp_v53.yaml`；中断后设置
`RERANK_RESUME=1` 即可续跑。

## VMP-v5.2 共享本地 vLLM 证据集合重排

V5.1 的 nearest-prototype promotion 在训练内达到 `88/94`，但
leave-one-question-out 仍为 `85/94`，因此没有通过 Dev gate，也没有运行 Test。
V5.2 放弃该高维原型分类器，改为两阶段链路：

```text
V4.3 / V5 memory adapter
  -> 各自生成 Top-30 session candidates
  -> 相同的 query-aware session excerpt
  -> 相同的本地 vLLM evidence-set prompt
  -> 保护原 Top-4，只开放第 5 个证据槽
  -> 相同的 Top-5 QA reader
```

reranker 看不到框架名称、gold session ID、gold answer 或 question type。它只接收
question、question date 和候选 session；所有方法统一使用
`Qwen/Qwen2.5-7B-Instruct`、temperature `0`、top-p `1`、相同 prompt 和候选数。
Dev gate 同时要求 V5.2 优于 raw V5 和经过相同 reranker 的 V4.3，避免把通用 LLM
增益错误归因给 VMP memory。

4090D 单卡建议严格分四阶段执行，避免 BGE-M3 与 vLLM 同时占用显存。

第一阶段：关闭 vLLM，生成 Dev Top-30 candidates：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=dev_candidates \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v52_experiment.sh
```

第二阶段：启动本地 vLLM，然后运行 Dev rerank 和质量门控：

```bash
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.90
export VMP_VLLM_ENABLE_TOOL_CALLING=0
uv run --no-sync bash scripts/serve_vllm.sh
```

在另一个终端运行：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=dev_rerank \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
uv run --no-sync bash scripts/run_vmp_v52_experiment.sh
```

如果 rerank 被中断，保持相同 run ID 并设置 `RERANK_RESUME=1` 即可断点续跑。只有
Dev gate 输出 `status=passed` 并生成
`outputs/longmemeval/gates/vmp_v52_seed42_dev_pass.json` 后，才能继续。

第三阶段：停止 vLLM，再生成密封 Test candidates：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=test_candidates \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
EMBEDDING_BATCH_SIZE=2 \
uv run --no-sync bash scripts/run_vmp_v52_experiment.sh
```

第四阶段：重新启动同一个 vLLM，然后运行 Test rerank、可选 QA 和论文表格：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
STAGE=test_rerank \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
RUN_QA=1 \
uv run --no-sync bash scripts/run_vmp_v52_experiment.sh
```

配置和公平性声明见 `configs/vmp_v52.yaml`。Dev/Test candidate、rerank、逐问题
LLM response、prompt hash、token usage、fallback 状态和运行日志都会分别保存；解析
失败时保持原排序，并由 gate 限制 fallback rate，绝不会静默使用 gold label 修正结果。

## VMP-v5.1 分层 Top-10 安全提升

VMP-v5.1 在 V5 的 session semantic、Top-1 turn semantic 和 turn BM25
分层融合之后，重新使用 Dev 数据拟合与新分数几何匹配的 promotion ranker。它严格
保留 hierarchical Top-10 候选集合和 Top-4，只有第 5 个槽位允许第 6--10 名挑战；
最终证据顺序仍由冻结的 VMP policy score 决定。

promotion margin 只根据 leave-one-question-out Dev 指标选择。模型工件同时保存
in-sample 指标、LOO 指标、完整 margin trials 和逐问题审计；论文 gate 只使用
LOO 指标，且至少要求相对 V4.3 和 promotion 前 V5 各多召回一个问题。gate
通过前不会访问 Test。

服务器运行：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_ID=lme_test_vmp_v51_seed42 \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
EMBEDDING_BATCH_SIZE=2 \
RUN_QA=0 \
uv run --no-sync bash scripts/run_vmp_v51_experiment.sh
```

默认使用 `GRID_STEP=0.05` 的 693 个融合 trial。已有 BGE-M3 SQLite cache
会被复用。主要工件为 `vmp_v51_seed42.json`、
`vmp_v51_seed42_search.json` 和 `vmp_v51_seed42_dev_audit.jsonl`。

## VMP-v5 分层 Session/Turn 检索

VMP-v5 用分层索引解决长 session 被单个 BGE-M3 向量截断或稀释的问题。每个
LongMemEval history session 同时建立 session chunk 和 role-prefixed turn
chunks；查询分别计算 session semantic、pooled turn semantic 和 turn BM25，
再使用只在固定 Dev split 搜索的三路权重聚合回官方 session ID。最终 Top-5
成员由 hierarchical dense head 冻结决定，排序继续复用 V4.3 的 VMP policy 和
非破坏 lifecycle。

V5 不复用 V4.3 的 nearest-prototype promotion ranker，因为新的分层分数改变了
特征几何；直接复用旧 prototype 会造成未校准分数。当前 `2.0` 工件只使用原始
turn，不调用 LLM，因此可以先独立验证分层检索贡献。后续 atomic-memory writer
必须通过同一个本地 vLLM endpoint 生成，并作为单独消融报告。

V5 只有在 Dev Recall-All@5、相对 V4.3 增量、turn 权重和 fold 稳定性门禁全部
通过后才会运行 Test：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_ID=lme_test_vmp_v5_seed42 \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
RUN_QA=0 \
uv run --no-sync bash scripts/run_vmp_v5_experiment.sh
```

V5 的 split provenance 使用语义指纹：数据集 SHA-256、question ID
集合、seed、划分策略以及完整 Dev/Test question assignment 都必须一致；
`created_at` 和不同机器上的绝对 `dataset_path` 不参与语义指纹。重复执行 split
脚本会复用已有的等价 manifest，不再改写时间戳。旧 V4.3 工件若仅文件 SHA-256
不同但语义划分完全一致，会记录 warning 后继续；真实的数据或划分变化仍会失败。

脚本会生成 `vmp_v5_seed42.json`、完整 grid-search report、Test retrieval
JSONL、运行日志和论文表格。V5 会额外预热 turn embeddings，正式计时阶段不会
把首次 embedding 下载或生成混入某个单独方法。

## VMP-v4.3 稳健 Dense 保护链路

VMP-v4 面向 V3 暴露出的泛化与延迟问题：完整保留 Dense Top-10 证据集合，
Top-5 至少保留四个原 Dense Top-5 项，并且 Dense 第 6--10 名只有超过 promotion
margin 才能进入 Top-5。查询无关 policy features 和 lifecycle 关系只预计算一次。
Dev 搜索除整体 Recall-All@5 外，同时约束确定性 fold 稳定性、问题类型宏平均召回
与最差类型召回。

搜索器使用与 `check_vmp_v4_gate.py` 完全相同的门禁阈值选择最终 trial，避免
fold 更稳定但绝对 Recall 不达标的 trial 覆盖可通过门禁的 trial。模型 metadata
同时记录 `max_dev_recall_all_at_5_seen` 和 `dev_oracle_ceiling_metrics`，用于区分
“搜索没有找到”与“Dense guard 结构理论上无法达到门禁”。

V4.1 的 512-trial 搜索计划还包含 36 个可审计的 Dev-only warm-start trial：
它们复用同一固定 Dev split 上 V3 已学到的权重，并重新测试 V4 Dense guard 下的
零/低 promotion margin。warm start 不会自动入选，仍需通过与随机 trial 完全相同
的 Dev 指标和质量门禁；Test 标签不参与参数生成或选择。搜索报告的
`parameter_source`、模型的 `selected_parameter_source` 和
`search_parameter_source_counts` 会记录完整来源。

V4.2 在全局参数搜索之后增加一个 Dev-only pairwise promotion ranker。它只在
Dense 第 5--10 名中学习单个开放 Top-5 槽位应保留第 5 名还是晋升候选，不改变
受保护的 Dense Top-4，也不改变 Dense Top-10 证据集合。模型同时记录训练拟合和
leave-one-question-out 指标；前者用于确认实现能达到 Dev guard oracle，后者用于
显式披露泛化风险。运行时不使用 LongMemEval 的 `question_type`，只使用查询文本
派生信号和统一的检索/记忆特征。V4.3 进一步将 promotion membership score 与
最终 evidence ordering 解耦：前者只决定第 5 个成员，后者始终使用冻结的 base
policy score，避免扰乱受保护的 Dense Top-4；prototype 距离在 NumPy 环境批量
向量化，并且只计算 Dense Top-10。

V4.3 工件 schema 为 `1.6`，旧 V3/V4.0–V4.2 工件必须重新训练。服务器入口命令：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_ID=lme_test_vmp_v43_seed42 \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
uv run --no-sync bash scripts/run_vmp_v4_experiment.sh
```

默认只运行 retrieval。只有共享 vLLM 服务已经启动且 GPU 显存充足时，才设置
`RUN_QA=1`。

VMP-MemOS 是一个面向长期 LLM Agent 的可解释 Memory Policy Layer。项目把“记忆内容是什么”和“记忆应该如何被管理”分开建模：`content_embedding` 表示内容语义空间，`policy_embedding` / `PolicyFeatures` 表示写入、更新、合并、归档、召回等管理信号。

当前实现保持论文实验的可控边界：不实现 Web UI、RL 或大型模型训练；toy benchmark 可以在没有 GPU 的环境先跑通；LongMemEval 论文 benchmark 已开始接入；LLM 能力通过可选 vLLM OpenAI-compatible server 接入，只在服务器上启动模型。

## 当前实现范围

已完成 `IMPLEMENTATION_PLAN.md` 中 Phase 0 至 Phase 10，并补齐 MVP 需要的 Hybrid backend：

- 标准 Python 3.11+ `src/` 项目结构；
- `pyproject.toml`、`.env.example`、`configs/default.yaml`；
- 核心 Pydantic schema：`Event`、`MemoryCandidate`、`MemoryItem`、`PolicyFeatures`、`MemoryOperation`、`RetrievalResult`、`BenchmarkSample`、`BenchmarkResult`；
- 可重复执行且不覆盖已有数据的 `scripts/init_workspace.py`；
- Markdown + YAML frontmatter 的 `FileMemoryBackend`；
- SQLite + embedding cache + cosine search 的 `VectorMemoryBackend`；
- Markdown source-of-truth + vector retrieval index 的 `HybridMemoryBackend`；
- `BaseEmbedder`、`SentenceTransformerEmbedder`、`SQLiteEmbeddingCache`、`CachedEmbedder`；
- 规则化 `PolicyFeatureBuilder`，可生成 16 个可解释 policy features；
- `RuleBasedPolicyController`，可计算 Write/Retrieve/Update/Merge/Archive/Compress score 并输出 operation decision；
- `MemoryOperationExecutor`，可执行 `ADD / UPDATE / MERGE / ARCHIVE / RETRIEVE / IGNORE` decision；
- toy benchmark runner 和 Markdown report 导出；
- Phase 8 baseline：`no_memory`、`full_context`、`summary_memory`、`naive_vector_rag`、`vector_rag_recency`、`vector_rag_importance`、`vmp_rule`；
- Phase 9 learned policy：从 toy benchmark / operation logs 构造训练样本，训练纯 Python multiclass logistic regression，输出 operation probabilities；
- Phase 10 ablation runner：支持禁用 `recency / contradiction / redundancy / success_contribution / token_cost` 并导出 Markdown 报告；
- vLLM LLM client：通过 OpenAI-compatible `/v1/chat/completions` 调用本地 vLLM 服务，并提供可选 LLM memory candidate extractor；
- LongMemEval-cleaned 接入骨架：loader、inspector、Event 转换、session-level evidence chunk；
- 论文实验 adapter 基类：`RetrievedMemory`、`BaseMemoryFrameworkAdapter`、`FrameworkRegistry`、framework controllability audit；
- 第一批可控 retrieval adapter：`empty`、`bm25`、`naive_vector`、`vector_recency`、`vector_importance`、`vmp_rule`；
- VMP-v4：固定 SHA-256 question-level dev/test split，保护 Dense Top-10 集合，使用受保护 Top-5 policy 重排、稳定性调参与缓存的非破坏生命周期；只有通过稳健 Dev 门禁后才允许访问 Test；
- VMP-v5：Dev-only session/turn/BM25 分层融合，turn embedding 聚合回官方 session provenance，冻结 hierarchical Top-5 后复用 VMP policy ordering；
- LongMemEval 消融：在同一冻结模型上运行 7 个 feature ablation 和 3 个 operation ablation，导出 retrieval/QA delta 表；
- Cost analysis：离线聚合 ingestion/retrieval/reader 延迟、token、active memory、storage 和每个正确答案成本；
- Case export：从 test 主实验与消融 run 中确定性导出四类可审计论文案例；
- LongMemEval retrieval runner：统一 BGE-M3 embedding、session/turn ingestion、官方 Recall-All/Any@5/10 与 NDCG、补充 fractional recall / MRR、延迟与存储统计；
- LongMemEval QA runner：统一 vLLM reader、固定 prompt、断点续跑、本地 QA metrics 和官方兼容 hypothesis；
- retrieval 论文表格导出：CSV、Markdown、LaTeX；
- 官方 Mem0 OSS adapter：固定 `mem0ai==2.0.10`，统一 vLLM/BGE-M3，支持 evidence provenance、workspace reset 和 smoke 凭证；
- 官方 LangMem adapter：固定 `langmem==0.0.30`，直接使用 memory store manager、共享 BGE-M3 embedder 和本地 vLLM；
- 官方 Graphiti adapter：固定 `graphiti-core==0.29.2`，统一 vLLM/BGE-M3，通过专用 Neo4j 保存知识图谱并导出 episode provenance；
- 官方 Letta adapter：固定 `letta-client==1.12.1` 与 Letta Server `0.16.8`，使用 agent-managed core/archival memory，并导出带 provenance 的 evidence；
- 基础 pytest 测试，覆盖 schema、workspace 初始化、文件后端、embedding cache、向量后端、hybrid 后端、policy feature builder、rule-based controller、operation executor、benchmark runner、learned policy、ablation runner、vLLM client config 和 LLM extraction parsing。

尚未实现：Web UI、RL 和官方框架全量正式服务器实验。

## 项目结构

```text
.
├── configs/
│   ├── default.yaml
│   ├── benchmark.yaml
│   └── llm.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
│       └── memory_policy_toy.jsonl
├── memory_workspace/
│   ├── INDEX.md
│   ├── MEMORY.md
│   ├── memories/
│   ├── archive/
│   ├── versions/
│   ├── vector/
│   ├── cache/
│   ├── logs/
│   └── {projects,skills,episodes,resources}/
├── outputs/
│   ├── runs/
│   ├── reports/
│   ├── figures/
│   └── models/
├── scripts/
│   ├── audit_frameworks.py
│   ├── create_longmemeval_split.py
│   ├── export_longmemeval_ablation.py
│   ├── export_longmemeval_cost.py
│   ├── export_longmemeval_cases.py
│   ├── download_longmemeval.py
│   ├── inspect_longmemeval.py
│   ├── init_workspace.py
│   ├── run_benchmark.py
│   ├── run_demo.py
│   ├── run_llm_smoke.py
│   ├── run_longmemeval_retrieval.py
│   ├── run_longmemeval_qa.py
│   ├── run_longmemeval_ablation.sh
│   ├── run_vmp_v4_experiment.sh
│   ├── run_vmp_tuned_experiment.sh
│   ├── check_vmp_v4_gate.py
│   ├── run_ablation.py
│   ├── serve_embeddings.py
│   ├── serve_graphiti_neo4j.sh
│   ├── serve_letta.sh
│   ├── serve_vllm.sh
│   ├── train_policy.py
│   ├── train_vmp_tuned.py
│   └── setup_server.sh
├── src/vmp_memos/
│   ├── backends/
│   ├── benchmark/
│   ├── extraction/
│   ├── frameworks/
│   ├── longmemeval/
│   ├── embeddings/
│   ├── llm/
│   ├── operations/
│   ├── policy/
│   └── schemas/
└── tests/
```

## 在 Linux / 4090D 服务器上运行

推荐一键 bootstrap：

```bash
git pull
bash scripts/setup_server.sh
```

等价手动命令：

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/init_workspace.py
python scripts/run_demo.py --backend file --workspace outputs/demo-workspace
python scripts/run_benchmark.py --config configs/benchmark.yaml
python scripts/train_policy.py --config configs/benchmark.yaml
python scripts/run_benchmark.py --config configs/benchmark.yaml --policy learned
python scripts/run_ablation.py --config configs/benchmark.yaml
python -m pytest
```

## vLLM LLM Integration

本项目的 LLM 接入方式是：在服务器上用 vLLM 启动 OpenAI-compatible API server，项目代码通过 HTTP 调用 `/v1/chat/completions`。本地开发不需要安装 vLLM，也不会自动下载模型。

推荐先在 4090D 服务器上安装基础依赖：

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

安装 vLLM：

```bash
python -m pip install vllm
```

启动 vLLM server。默认模型是 `Qwen/Qwen2.5-7B-Instruct`，你可以按显存情况替换：

```bash
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.90
bash scripts/serve_vllm.sh
```

服务启动后，另开一个 shell：

```bash
source .venv/bin/activate
export VMP_LLM_API_KEY="local-vllm-key"
python scripts/run_llm_smoke.py \
  --config configs/llm.yaml \
  --prompt "用一句话说明 VMP-MemOS 的作用。"
```

如果没有设置 vLLM API key，也可以不导出 `VMP_LLM_API_KEY`，但 `scripts/serve_vllm.sh` 和 client 两边要保持一致。

运行 LLM memory candidate extraction smoke：

```bash
python scripts/run_llm_smoke.py \
  --config configs/llm.yaml \
  --extract-memory \
  --scope project/vmp-memos \
  --prompt "用户现在主攻 Agent 和 LLM 应用开发，不再 all in Java 后端。"
```

常用覆盖参数：

```bash
python scripts/run_llm_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-key local-vllm-key \
  --max-tokens 256 \
  --temperature 0
```

如果要运行向量检索或 hybrid demo，请安装 embedding extra：

```bash
python -m pip install -e ".[dev,embeddings]"

python scripts/run_demo.py \
  --backend vector \
  --workspace outputs/demo-vector-workspace \
  --device cuda

python scripts/run_demo.py \
  --backend hybrid \
  --workspace outputs/demo-hybrid-workspace \
  --device cuda
```

也可以让 bootstrap 脚本安装 embedding extra：

```bash
VMP_EXTRAS="dev,embeddings" bash scripts/setup_server.sh
```

如需指定 SentenceTransformers 模型缓存目录：

```bash
python scripts/run_demo.py \
  --backend vector \
  --workspace outputs/demo-vector-workspace \
  --device cuda \
  --model-cache-dir /path/to/model-cache
```

## LongMemEval 论文 benchmark 接入

当前已完成 LongMemEval-cleaned 的数据接入、schema 转换、可控 retrieval adapters、retrieval runner 与统一 vLLM QA runner。确定性检索 baseline 不调用 LLM；官方 memory frameworks 可在记忆抽取阶段调用同一个本地 vLLM，所有向量方法统一使用 BGE-M3。

下载 LongMemEval-cleaned：

```bash
source .venv/bin/activate
python scripts/download_longmemeval.py --target data/longmemeval
```

如果服务器网络无法直接访问 Hugging Face，也可以手动从 `xiaowu0162/longmemeval-cleaned` 下载以下文件并放入 `data/longmemeval/`：

```text
longmemeval_oracle.json
longmemeval_s_cleaned.json
longmemeval_m_cleaned.json
```

检查数据：

```bash
python scripts/inspect_longmemeval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --limit 3
```

安装检索实验依赖：

```bash
python -m pip install -e ".[dev,embeddings]"
```

先做不下载 embedding 模型的结构 smoke：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vmp_rule \
  --top-k 5 \
  --retrieval-depth 10 \
  --limit 20 \
  --no-embeddings \
  --run-id lme_retrieval_smoke
```

在 4090D 上运行正式 retrieval 实验：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --top-k 5 \
  --retrieval-depth 10 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir /path/to/huggingface-cache \
  --embedding-cache-db outputs/longmemeval/cache/bge_m3.sqlite3 \
  --embedding-batch-size 1 \
  --prewarm-embeddings \
  --run-id lme_s_bge_m3_main
```

`--top-k 5` 是后续 QA 使用的 evidence 数量；`--retrieval-depth 10` 会额外保存前 10 条结果，以便计算官方 Recall-All/Any@10。4090D 默认使用 `--embedding-batch-size 1`，避免 BGE-M3 对长 session 编码时 OOM。持久缓存会在计时前统一预热，并把 batch size、缓存路径、预热耗时、hit/miss/generated 写入 manifest，保证方法顺序不会污染 latency。每个 run 会输出：

```text
outputs/longmemeval/runs/{run_id}/manifest.json
outputs/longmemeval/runs/{run_id}/summary.json
outputs/longmemeval/runs/{run_id}/{method}/retrieval.jsonl
outputs/longmemeval/runs/{run_id}/{method}/summary.json
```

正式结果不要使用 `--no-embeddings`。该选项只用于验证数据、runner 和输出格式。

### VMP-v4.3：固定 dev/test、稳健门禁与 Dense 集合安全重排

当前链路不再允许 policy 在检索时删除 source session：

```text
LongMemEval sessions
  -> BGE-M3 dense ranking
  -> immutable Dense Top-10 safety set
  -> cached query-independent PolicyFeatures + BM25 anchor
  -> temporal-intent gated PolicyFeatures
  -> guarded Top-5 selection (at least four Dense Top-5 items)
  -> cached non-destructive active/superseded/duplicate annotation
  -> Top-10 retrieval evidence with the same set as Dense Top-10
  -> Top-5 shared vLLM reader context
```

第 0 个 Dev trial 固定为纯 dense 安全基线。其余 trial 使用受限的
`policy_adjustment_limit` 和 promotion margin，并同时评估确定性 fold 稳定性、
question-type 宏平均和最差类型召回。冻结模型必须同时满足绝对 Dev 指标、
相对 dense 增益和稳健性门禁，否则
脚本在访问 Test 前退出。

先生成一次固定 split。分配只使用 `question_id` 和 seed，不读取答案；manifest
同时记录原始数据 SHA-256，后续数据文件有任何字节变化都会拒绝运行：

```bash
python scripts/create_longmemeval_split.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --output outputs/longmemeval/splits/dev_test_seed42.json \
  --seed 42 \
  --dev-size 100 \
  --test-size 400
```

只用 dev 的 gold session 调优并冻结模型：

```bash
python scripts/train_vmp_tuned.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --output outputs/longmemeval/models/vmp_v43_seed42.json \
  --report outputs/longmemeval/models/vmp_v43_seed42_search.json \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir "${HOME}/.cache/huggingface" \
  --embedding-cache-db outputs/longmemeval/cache/bge_m3.sqlite3 \
  --embedding-batch-size 8 \
  --trials 512 \
  --stability-folds 5 \
  --tuning-seed 2025
```

模型工件记录 dataset SHA、split manifest SHA、embedding identifier、搜索 seed、
目标函数、dev 指标和 `test_labels_used=false`。正式评测必须显式指定 test；runner
会拒绝在训练 split 上运行 `vmp_tuned`，也会拒绝数据、split 或 embedding 不匹配：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --split test \
  --vmp-tuned-model outputs/longmemeval/models/vmp_v43_seed42.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned \
  --top-k 5 \
  --retrieval-depth 10 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir "${HOME}/.cache/huggingface" \
  --embedding-cache-db outputs/longmemeval/cache/bge_m3.sqlite3 \
  --embedding-batch-size 8 \
  --prewarm-embeddings \
  --run-id lme_test_vmp_v43_seed42
```

一条脚本可顺序执行 split、dev 调优、test retrieval 和表格导出：

```bash
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
RUN_ID=lme_test_vmp_v43_seed42 \
uv run --no-sync bash scripts/run_vmp_v4_experiment.sh
```

脚本将表格写入 `outputs/longmemeval/tables/{RUN_ID}/`，避免新 run
覆盖旧实验表格；控制台和 Python 日志同时保存在
`outputs/longmemeval/logs/{RUN_ID}.log`。

脚本默认不启动或调用 vLLM，避免单卡上 vLLM 预分配显存后 BGE-M3 无法加载。
retrieval 完成后再启动 vLLM 并执行下文 QA 命令。若你已为两者留出足够显存，
可以设置 `RUN_QA=1` 让脚本继续执行 QA。

runner 会拒绝任何非空但无法解析的日期。官方 LongMemEval
`YYYY/MM/DD (Day) HH:MM` 与 ISO-8601 都受支持；`question_id` 以 `_abs`
结尾的 30 条题目按官方定义视为 abstention：retrieval 默认跳过，QA 单独计算
abstention accuracy。

### LongMemEval 消融实验

消融实验严格复用前一步生成的 split manifest、BGE-M3 和冻结
`vmp_v43_seed42.json`，不会为任何消融变体重新调参。
V4.3 模型 schema 为 `1.6`，记录 Dense Top-10 安全集合、受保护 Top-5 重排、
Dev pairwise promotion ranker、独立的 base-policy ordering、Dev safety baseline
和非破坏 lifecycle policy；拉取新代码后应先重新执行
`run_vmp_v4_experiment.sh`，旧 `1.0`–`1.5` 工件会被明确拒绝。变体包括：

```text
VMP-full
VMP w/o recency
VMP w/o contradiction
VMP w/o redundancy
VMP w/o importance
VMP w/o confidence
VMP w/o token_cost
VMP w/o scope_match
VMP w/o update_operation
VMP w/o merge_operation
VMP w/o archive_operation
```

operation ablation 具有独立语义：update 只在问题具有时间/更新意图时参与重排；
archive 和 merge 只写入 `superseded/duplicate` 状态并施加有上限的软惩罚，
不会在 retrieval 阶段删除原始 session。每道题的 operation counts、状态数量、
禁用组件和模型 split ID 都写入 retrieval record。
每道题的 operation counts、禁用组件和模型 split ID 都写入 retrieval record。

先在未启动 vLLM 时完成全部 test retrieval。脚本使用持久化 SQLite embedding
cache，避免 11 个变体重复计算相同 BGE-M3 向量：

```bash
RUN_ID=lme_test_ablation_seed42 \
bash scripts/run_longmemeval_ablation.sh
```

启动 vLLM 后，为同一 retrieval run 生成统一 QA：

```bash
ABLATION_METHODS="vmp_tuned,vmp_tuned__no_recency,vmp_tuned__no_contradiction,vmp_tuned__no_redundancy,vmp_tuned__no_importance,vmp_tuned__no_confidence,vmp_tuned__no_token_cost,vmp_tuned__no_scope_match,vmp_tuned__no_update_operation,vmp_tuned__no_merge_operation,vmp_tuned__no_archive_operation"

python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_ablation_seed42 \
  --methods "${ABLATION_METHODS}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128

python scripts/export_longmemeval_ablation.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_ablation_seed42
```

输出 `table4_ablation.{csv,md,tex}`，包含官方 Recall-All@5、MRR、NDCG@5、
retrieved tokens、Normalized EM、Token F1 及其相对 VMP-full 的差值。

### Cost and Efficiency

成本分析完全读取已有 retrieval/QA JSONL，不会重新运行 embedding 或 LLM。
默认要求 QA 已完整结束，并验证每个方法的 question ID 和顺序与 retrieval
严格一致：

```bash
python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v43_seed42
```

输出：

```text
outputs/longmemeval/tables/cost_analysis.json
outputs/longmemeval/tables/table5_cost.csv
outputs/longmemeval/tables/table5_cost.md
outputs/longmemeval/tables/table5_cost.tex
```

Table 5 包含 mean/P95 ingestion、retrieval、reader、end-to-end latency，
retrieved/reader/framework tokens、active memory count、memory retention ratio、
storage size、operation counts，以及 observed tokens / milliseconds per correct
answer。本地 vLLM 不伪造美元价格；官方框架无法导出内部 LLM usage 时保留
`null` 并报告 coverage，绝不按零成本处理。

如果只想在 QA 前预览 retrieval 成本，可以显式运行：

```bash
python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v43_seed42 \
  --allow-missing-qa
```

### 论文案例导出

案例导出需要一个包含 `naive_vector`、`vmp_tuned` 及完整 QA 的主实验 run，
以及一个包含 `vmp_tuned__no_archive_operation` 的消融 run：

```bash
python scripts/export_longmemeval_cases.py \
  --retrieval-run outputs/longmemeval/runs/lme_official_main \
  --ablation-run outputs/longmemeval/runs/lme_test_ablation_seed42 \
  --vmp-method vmp_tuned \
  --vector-method naive_vector
```

导出器会验证两个 run 使用相同 dataset SHA、test split 和冻结 VMP 模型，
然后确定性选择：

```text
1. VMP 正确处理 knowledge update
2. NaiveVectorRAG 召回与新证据相关但更旧的非 gold 证据
3. VMP archive 在保留 source evidence 的同时降低 superseded memory 排名
4. VMP 的真实错误案例
```

输出：

```text
outputs/longmemeval/cases/manifest.json
outputs/longmemeval/cases/cases.json
outputs/longmemeval/cases/paper_cases.md
outputs/longmemeval/cases/case1_knowledge_update.json
outputs/longmemeval/cases/case2_stale_vector_retrieval.json
outputs/longmemeval/cases/case3_archive_suppression.json
outputs/longmemeval/cases/case4_vmp_error.json
```

JSON 保留 gold sessions、retrieved evidence、QA prediction、policy
features/contributions、operation counts 和源 manifest hash。Markdown
只使用确定性模板，不额外调用 LLM；若没有符合某类定义的样本会报错，不会用
“最接近”的样本冒充成功或失败案例。

导出 retrieval 论文表格：

```bash
python scripts/export_longmemeval_tables.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main
```

输出包括：

```text
outputs/longmemeval/tables/table1_retrieval_overall.{csv,md,tex}
outputs/longmemeval/tables/table2_by_question_type.{csv,md,tex}
```

启动统一 vLLM reader 后运行端到端 QA：

```bash
export VMP_LLM_API_KEY="local-vllm-key"

python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128
```

如果服务器任务中断，使用完全相同的参数并添加 `--resume`。runner 会跳过已经写入的 question，并检查整个 run 只能出现一个 reader provider/model：

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128 \
  --resume
```

QA 输出：

```text
outputs/longmemeval/runs/{run_id}/qa/manifest.json
outputs/longmemeval/runs/{run_id}/qa/{method}.jsonl
outputs/longmemeval/runs/{run_id}/qa/{method}.summary.json
outputs/longmemeval/runs/{run_id}/qa/summary.json
outputs/longmemeval/runs/{run_id}/hypotheses/{method}.jsonl
```

本地 QA 指标包括 Normalized Exact Match、Token F1、Contains Answer 和 Abstention Accuracy，同时记录 reader token usage 与端到端延迟。`hypotheses/{method}.jsonl` 使用 `{"question_id": "...", "hypothesis": "..."}` 格式，可交给官方 evaluator；官方 GPT-based evaluator 不属于本地 pipeline 的硬依赖。

审计外部官方框架是否能公平进入主表：

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024
```

审计输出：

```text
outputs/longmemeval/audit/framework_controllability.json
outputs/longmemeval/tables/table6_fairness.csv
```

当前策略是：官方 OSS adapter 优先；只有通过本地 vLLM、本地 embedding、evidence export、workspace reset 等可控性检查的方法，才允许进入论文主性能表。style baseline 只作为兜底和附录，不冒充官方框架结果。

### 官方 Mem0 / LangMem / Graphiti / Letta OSS 对比

Mem0 adapter 直接调用官方 `mem0.Memory`，没有复刻 Mem0 算法。为了避免框架版本漂移，论文环境固定使用 `mem0ai==2.0.10`：

```bash
python -m pip install -e \
  ".[dev,embeddings,official-mem0,official-langmem,official-graphiti,official-letta]"
```

Mem0 ingestion 会调用本地 vLLM 做官方 memory extraction。Qwen2.5 的工具调用需要 Hermes parser；`scripts/serve_vllm.sh` 已默认启用。由于 vLLM 与 BGE-M3 同时占用 4090D，建议 Mem0 实验降低 vLLM 显存预留：

```bash
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VMP_VLLM_ENABLE_TOOL_CALLING=1
export VMP_VLLM_TOOL_CALL_PARSER=hermes
bash scripts/serve_vllm.sh
```

四个官方 adapter 共用 `official_llm_temperature=0.0` 和
`official_llm_max_tokens=512`；实际值会写入 retrieval manifest。若修改，
必须对所有官方框架使用同一组 CLI 参数或环境变量。

先在另一个 shell 运行官方 adapter smoke：

```bash
source .venv/bin/activate
export VMP_LLM_API_KEY="local-vllm-key"

python scripts/run_official_framework_smoke.py \
  --framework mem0 \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda
```

使用同一套模型验证 LangMem：

```bash
python scripts/run_official_framework_smoke.py \
  --framework langmem \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda
```

Graphiti 还需要一个只供本项目使用的 Neo4j。下面的脚本会创建带专用标签的容器；benchmark 每道题都会清空该数据库的全部节点，因此绝不能把 URI 指向已有业务数据库：

```bash
export VMP_GRAPHITI_NEO4J_PASSWORD="replace-with-a-strong-password"
export VMP_GRAPHITI_NEO4J_URI="bolt://127.0.0.1:7687"
export VMP_GRAPHITI_NEO4J_USER="neo4j"
bash scripts/serve_graphiti_neo4j.sh
```

使用同一套 vLLM 与 BGE-M3 验证 Graphiti。`--graphiti-allow-destructive-reset` 是有意设置的安全确认：

```bash
python scripts/run_official_framework_smoke.py \
  --framework graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda \
  --graphiti-neo4j-uri bolt://127.0.0.1:7687 \
  --graphiti-allow-destructive-reset
```

Letta 是独立的官方状态服务。先在新 shell 启动同一个 BGE-M3 的
OpenAI-compatible embedding endpoint：

```bash
source .venv/bin/activate
python scripts/serve_embeddings.py \
  --host 0.0.0.0 \
  --port 8001 \
  --model BAAI/bge-m3 \
  --device cuda
```

再启动固定为 `0.16.8` 的专用 Letta Server。该脚本使用 Linux host
network，使容器连接前面的 vLLM 和 embedding endpoint：

```bash
export VMP_LLM_BASE_URL=http://127.0.0.1:8000/v1
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LETTA_BASE_URL=http://127.0.0.1:8283
export VMP_LETTA_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
bash scripts/serve_letta.sh
```

最后验证 Letta agent-managed memory。Letta 会为每道题创建独立 agent，
并在下一题或退出时删除：

```bash
python scripts/run_official_framework_smoke.py \
  --framework letta \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --letta-base-url http://127.0.0.1:8283 \
  --letta-embedding-base-url http://127.0.0.1:8001/v1
```

Letta 的旧 `vllm` provider 会覆盖生成上限，因此本适配器使用其官方
OpenAI-compatible provider 连接同一个本地 vLLM。这样四个框架才能严格
共享温度、生成上限、API key 和模型；请求不会流向 OpenAI 云端。

smoke 成功后重新执行审计：

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --official-llm-max-tokens 512 \
  --official-llm-temperature 0 \
  --verification-dir outputs/longmemeval/audit
```

只有客户端/服务端版本、模型配置和 smoke 凭证都匹配时，`mem0` /
`langmem` / `graphiti` / `letta` 才会得到
`main_table_eligible=true`。先跑 20 条：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --split test \
  --vmp-tuned-model outputs/longmemeval/models/vmp_v43_seed42.json \
  --methods bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned,mem0,langmem,graphiti,letta \
  --top-k 5 \
  --retrieval-depth 10 \
  --limit 20 \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --graphiti-neo4j-uri bolt://127.0.0.1:7687 \
  --graphiti-allow-destructive-reset \
  --letta-base-url http://127.0.0.1:8283 \
  --letta-embedding-base-url http://127.0.0.1:8001/v1 \
  --run-id lme_official_smoke20
```

确认 smoke 结果后移除 `--limit 20`，并更换新的 `--run-id`
运行完整 test split。正式官方框架实验必须保留相同 split 和冻结的
VMP-Tuned 模型，同时保留默认 `--official-memory-infer`；不能使用
`--no-embeddings` 或 `--no-official-memory-infer`。

retrieval 完成后，所有方法继续共用同一个 QA reader：

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_official_main \
  --methods bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned,mem0,langmem,graphiti,letta \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128
```

## Workspace 初始化

默认命令会创建缺失目录和空 JSONL 日志，同时保留已有 Markdown 与日志内容：

```bash
python scripts/init_workspace.py
```

可指定独立实验目录：

```bash
python scripts/init_workspace.py --workspace /path/to/experiment/memory_workspace
```

只有明确需要刷新种子 Markdown 时才使用 `--force`；即使使用该参数，已有 JSONL 日志也不会被截断。

## 后端说明

`FileMemoryBackend` 将活动记忆写到 `memories/{memory_id}.md`，归档记忆写到 `archive/{memory_id}.md`，更新前版本写到 `versions/{memory_id}/v000001.md`。每次 ADD / UPDATE / ARCHIVE / RETRIEVE 都会写入 `logs/operations.jsonl` 或 `logs/retrievals.jsonl`。Phase 2 的 file search 是确定性词法检索，便于早期验证。

`VectorMemoryBackend` 将当前记忆、content embedding 和版本历史写入 `vector/memories.sqlite3`，embedding cache 写入 `cache/embeddings.sqlite3`。它会记录 embedding namespace 和维度，防止同一个向量库混用不同模型。检索使用 query embedding 与候选 memory embedding 的 cosine similarity。

`HybridMemoryBackend` 将 `FileMemoryBackend` 作为 source of truth，将 `VectorMemoryBackend` 作为 retrieval index。`get/list` 从 Markdown 文件读取；`search` 先由向量后端排序，再用 file 后端按 ID hydrate 可读 memory；`add/update/archive` 会同步两个组件。

## Policy Feature Builder

`PolicyFeatureBuilder` 会生成 16 个 policy features：

```text
semantic_relevance, importance, confidence, recency,
stability, novelty, redundancy, contradiction,
staleness, access_frequency, success_contribution, failure_contribution,
token_cost, scope_match, actionability, privacy_risk
```

第一版完全规则化：如果调用方已经提供 embedding，则使用 cosine similarity；否则使用词法相似度、时间半衰期、scope、访问次数和关键词启发式。它不会调用 LLM，也不会依赖 GPU。

## Rule-based Policy Controller

`RuleBasedPolicyController` 实现六个规则分数：

```text
WriteScore, RetrieveScore, UpdateScore,
MergeScore, ArchiveScore, CompressScore
```

每个 `score_*()` 会返回分数、阈值、是否通过、贡献项和可读 reason；每个 `decide_*()` 会返回 `ADD / UPDATE / MERGE / ARCHIVE / COMPRESS / RETRIEVE / IGNORE` decision。Decision 可以转换为 `MemoryOperation`，供 executor 写入 JSONL。

## Operation Executor

`MemoryOperationExecutor` 将 policy decision 执行到任意 `BaseMemoryBackend` 上。第一版实现：

- `ADD`：写入新 memory；
- `UPDATE`：对目标 memory 应用 patch；
- `MERGE`：更新主 memory，并可归档 source memories；
- `ARCHIVE`：归档目标 memory；
- `RETRIEVE`：调用 backend search；
- `IGNORE`：不修改 backend，但写入一条 `IGNORE` operation log。

## Toy Benchmark Runner

默认数据集：

```text
data/benchmarks/memory_policy_toy.jsonl
```

覆盖 8 类场景：

```text
preference update, fact conflict, project state change,
multi-session integration, stale memory archive,
duplicate memory merge, long tool log compression,
failure-to-procedural-memory
```

默认 baseline：

```text
no_memory
full_context
summary_memory
naive_vector_rag
vector_rag_recency
vector_rag_importance
vmp_rule
```

其中 `vector_rag` 是 `naive_vector_rag` 的 CLI 兼容别名，方便按计划中的验收命令运行。

运行完整 toy benchmark：

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

只跑部分 baseline：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark.yaml \
  --baselines no_memory,vector_rag,vmp_rule \
  --run-id local_smoke
```

输出：

```text
outputs/runs/{run_id}/results.jsonl
outputs/reports/{run_id}.md
```

报表包含：

```text
Accuracy
Evidence Precision / Recall
Operation Recall
Write Precision
Memory Growth
Token Cost
Conflict Retrieval Rate
Stale Memory Usage Rate
Latency
```

## Learned Policy

Phase 9 的第一版 learned policy 是一个无额外依赖的 multiclass logistic regression。它输入 `PolicyFeatures` 的 16 维向量，输出以下 operation probabilities：

```text
ADD, UPDATE, MERGE, ARCHIVE, RETRIEVE, IGNORE
```

训练命令：

```bash
python scripts/train_policy.py --config configs/benchmark.yaml
```

默认输出：

```text
outputs/models/learned_policy.json
outputs/models/learned_policy_examples.jsonl
```

可叠加 operation logs 作为训练数据：

```bash
python scripts/train_policy.py \
  --config configs/benchmark.yaml \
  --operation-log memory_workspace/logs/operations.jsonl
```

运行 learned policy benchmark：

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml --policy learned
```

指定模型路径：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark.yaml \
  --policy learned \
  --policy-model-path outputs/models/learned_policy.json
```

## Ablation Runner

Phase 10 提供 feature-level 消融实验，用来观察 policy feature 对 VMP rule baseline 的影响。默认会比较：

```text
no_memory
naive_vector_rag
vmp_rule
vmp_rule__no_recency
vmp_rule__no_contradiction
vmp_rule__no_redundancy
vmp_rule__no_success_contribution
vmp_rule__no_token_cost
```

运行默认消融：

```bash
python scripts/run_ablation.py --config configs/benchmark.yaml
```

只跑指定 feature 消融：

```bash
python scripts/run_ablation.py \
  --config configs/benchmark.yaml \
  --disable recency \
  --disable contradiction
```

默认输出：

```text
outputs/runs/{run_id}/results.jsonl
outputs/reports/ablation.md
```

报告包含实验设置、baseline 对比、ablation 对比、完整指标表、错误案例和 memory operation 示例。

## 验证命令

```bash
python -m pytest
python -m pytest tests/test_ablation_runner.py
python -m pytest tests/test_benchmark_runner.py
python -m pytest tests/test_learned_policy.py
python -m pytest tests/test_hybrid_backend.py
python -m pytest tests/test_operation_executor.py
python -m pytest tests/test_policy_controller.py
python -m pytest tests/test_policy_feature_builder.py
python -m pytest tests/test_file_backend.py tests/test_vector_backend.py
python -m pytest tests/test_retrieval_metrics.py tests/test_longmemeval_retrieval_runner.py
python -m pytest tests/test_longmemeval_splits.py tests/test_vmp_tuned.py
python -m pytest tests/test_longmemeval_ablation.py
python -m pytest tests/test_longmemeval_cost.py
python -m pytest tests/test_longmemeval_cases.py
python -m pytest tests/test_qa_metrics.py tests/test_longmemeval_qa_runner.py
python -m pytest tests/test_mem0_official_adapter.py tests/test_framework_audit.py
python -m pytest tests/test_langmem_official_adapter.py
python -m pytest tests/test_graphiti_official_adapter.py
python -m pytest tests/test_letta_official_adapter.py tests/test_embedding_server.py
ruff check src scripts tests
mypy src scripts
```

## 快速示例

```python
from vmp_memos.backends import VectorMemoryBackend
from vmp_memos.policy import PolicyFeatureBuilder
from vmp_memos.schemas import MemoryItem, MemorySource

backend = VectorMemoryBackend("memory_workspace")
memory = MemoryItem(
    type="semantic",
    scope="career/agent-dev",
    content="用户当前主攻 Agent 开发和 LLM 应用开发。",
    source=MemorySource(source_type="conversation"),
)

memory = PolicyFeatureBuilder().enrich_memory(memory)
backend.add(memory, reason="New stable preference.")
results = backend.search("Agent 长期记忆开发", top_k=5)
```

所有本地敏感配置都应通过 `.env` 或 `configs/` 注入。不要提交 `.env`、API key、模型缓存或服务器绝对路径。
