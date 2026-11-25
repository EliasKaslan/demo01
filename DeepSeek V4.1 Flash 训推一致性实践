# DeepSeek V4.1 Flash 训推一致性实践

截至 2026 年 11 月 25 日上午，**40 层、每层 384 个路由专家的 BF16 模型，已在 8 机 64 张 Ascend 950DT 上完成 CountBenchQA 在线 GRPO 验收**。推理用 vLLM Ascend，训练用 verl 和 FSDPTurbo。全量长序列基线连续更新 28 步，共 1,792 条回答、3,670,016 个有效输出 token。对每个实际生成的 token，推理返回的 logprob 与训练 loss 使用的 logprob 都是有限 FP32，逐位相同。

## 1. 这个“零位差”怎么测

vLLM 在生成时保存回答的 token ID 和所选 token 的 logprob。训练 Actor 使用**同一版权重、同一张图、相同的前文和回答 ID**，对整段回答做带梯度前向。在真实策略 loss 入口，按因果位置将两侧对齐，用 `response_mask` 选出有效输出，逐个比较 FP32 的 32 位表示。旧策略字段在 bypass 模式下可以直接引用 rollout 分数，因此比较的一端必须取自 Actor 本次前向。

每步还核对反向、优化器更新以及推理 worker 收到的权重，下一步用更新后的策略重新生成。R15 在初始同步和 28 次更新后，共核对 **33,408 对敏感 FP32 权重哈希**（64 个 worker × 18 项 × 29 个版本）。这是所选 token 的前向分数验收，参数同步另有抽检；梯度检查关注反向是否真实执行、参数是否变化。在 R19 负对照中，独立验收器在一个有效 logprob 翻转一位后能报错。[训练评分](https://gitcode.com/EliasKaslan/dsv41-verl/blob/17fe22595de5d6112e0d3ec8f579f31bf5c2e28a/verl/utils/dsv41_logprob.py) · [验收代码](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/blob/8d91b3356a33e8580259978cf1fc551502faf9e7/experiments/2026-11-18-countbenchqa/tools/verify_training.py)

| 权重与运行 | 训练和生成条件 | 实测范围 | 有效 FP32 分数 / 位差数 |
|---|---|---:|---:|
| 全量 R7，自然结束 | 保留模型生成的结束位置 | 2 步，128 条回答 | 2,500 / 0 |
| 全量 R15，长序列基线 | 每条 1,024 个有效输入、2,048 个实际采样输出 | 28 步，1,792 条回答 | 3,670,016 / 0 |
| 全量 R19，性能配方 | 同一长序列负载，启用 CP 本地评分等优化 | 12 步，768 条回答 | 1,572,864 / 0 |
| 历史 Top32，单机八卡 | 40 层、每层保留 32 个路由专家，训练 CP1 | 100 步，9,600 条回答 | 1,885,622 / 0 |

早期定位先固定生成历史，对照单请求、混批、换序以及重新 prefill，再扩到 Top32 百步训练。推理 PCP8 与 TP8 分别另做过两步验收。全量八机采用 DP64 / EP64。

全量长序列实验每步 16 道题、每题采样 4 条回答。输入保留原图和计数问题，补充计数说明后由处理器核到 1,024 个有效 token；输出关闭 EOS 提前终止，实际生成 2,048 步。这组数据检验长历史上的数值和性能，自然回答的长度与质量应使用自然停止实验评价。[历史 Top32 百步验收](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/blob/8d91b3356a33e8580259978cf1fc551502faf9e7/experiments/2026-11-19-countbenchqa-performance/evidence/night22-long/optimized100.json)

## 2. 全量模型怎样分到 64 张卡

| | 全量验收配置 | 一条请求如何执行 |
|---|---|---|
| 推理 rollout | TP1 / DP64 / EP64，PCP1 / PP1，异步调度、`FULL_DECODE_ONLY` | 单条请求的注意力按原序列处理。384 个专家分到 64 个 EP rank，每 rank 持有 6 个，每个 token 选 6 个专家。 |
| 训练 Actor | TP1 / FSDP64 / EP64，KV-AllGather CP4，BSND | 16 个 CP 组，每组 4 个 rank 处理一条序列的不同段。每个 rank 保留本段 Q，汇集注意力所需的历史 KV。 |

训练从 meta 张量开始构造路由专家，等 EP 归属确定后加载各 rank 的权重切片，避免每个进程先分配完整专家权重。全量配置恢复了检查点中的两个 Engram 层（层号 1、14）。40 层策略网络、完整专家集合和两层 Engram 都参与前向；视觉编码器与两张主机侧 Engram 查找表冻结。推理与训练分阶段复用设备，rollout 进入一级休眠后释放设备内存，再执行 Actor 前反向。

## 3. 打开 BI 后，哪些算子会换

**BI（batch invariance，批次不变性）的目标**是同一条请求在单独运行、混批或换序时，对同一前文算出相同结果。推理服务启用 RL 配置中的 `enable_batch_invariant` 后设置 `VLLM_BATCH_INVARIANT=1`。Ascend worker 初始化时注册下表中的 NPU 实现。R19 的归档配置为 `enable_batch_invariant: true`，运行日志里有 64 条 Ascend worker 启用 BI 的记录。**源码记录的是分派规则。设备 trace 才能确认具体 kernel 的调用次数。** [RL 配置入口](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/ascend_config.py) · [完整注册与回退条件](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/batch_invariant.py)

| 调用入口 | 启用 BI 后的行为 | 生效条件 |
|---|---|---|
| `aten::mm`、`aten::matmul` | 优先调用 AscendC 的 `npu_mm_batch_invariant`、`npu_matmul_batch_invariant` | 可导入 `batch_invariant_ops` 时优先。只有 Triton 可用时，注册 Triton 版本，并额外注册 `aten::linear`。 |
| `aten::addmm`、`aten::bmm`、`aten::softmax`、`aten::_softmax` | 分别注册 Triton 的 `addmm_batch_invariant`、`bmm_batch_invariant`、`softmax_batch_invariant` | Triton 可用时注册，包括同时装有 AscendC 扩展的情况。 |
| `torch_npu.npu_fused_infer_attention_score` | 函数入口指向 `npu_fused_infer_attention_score_batch_invariant` | AscendC 扩展可用。V4.1 的 BF16 稀疏注意力另有专用入口，见下一节。 |
| `torch_npu.npu_add_rms_norm` | 先算残差加法，再调用 `npu_rms_norm`，同时返回加法结果 | AscendC 扩展可用。替换针对这个融合入口。 |
| `torch.sum`、`Tensor.sum` | 合适的 NPU 尾轴归约转入 `npu_reduce_sum_batch_invariant` | 输入为 FP16、BF16 或 FP32，满足单尾轴、默认输出类型等条件。显式 `dtype`、`out`、非尾轴等情况走原生求和。 |

同一初始化还设置确定性通信和权重布局（HCCL strict、LCCL 确定性、`weight_nz_mode=0`），保留调用方已设的 NPU 确定性等级 3。**模型内部另有条件选路**，几处最容易与上表混淆：

- **mHC。** BI 让 `hc_pre` 进入批次不变的前处理封装，训练同款 mHC 开关再选择 delayed-premix 算子 `mhc_pre_sinkhorn_premix`。`hc_post` 则直接受 `VLLM_ASCEND_DSV41_TRAINING_MHC` 控制，走训练同款 `mhc_post`。小批推理在已验证的形状和精度下最低补到 16 行；训练保留原分块和反向。[模型分派](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/models/deepseek_v41/model.py) · [mHC 实现](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/ops/batch_invariant_hc.py)
- **MoE。** 在本配方的通信选路中，BI 联合 `VLLM_ASCEND_DSV41_TRAINING_MOE_COMBINE` 固定 AllGather 分派。各 rank 把未加权的 BF16 专家结果放回对应 top-k 槽，清零无效槽，以整数位模式做 ReduceScatter，再按训练顺序用 FP32 权重加权。这样，短 decode 和长 prefill 能采用同一种合并算序。[通信选路](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/ascend_forward_context.py) · [专家槽合并](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/ops/dsv41_graph_moe.py)
- **Engram。** 全量模型的门控点积必须按训练顺序做尾轴求和。服务会替换 `Tensor.sum`，因此该调用点显式使用 `torch.ops.aten.sum.dim_IntList`。用真实权重对照，原融合门控在层 1、14 各比较 368,640 个 BF16 元素，分别有 4、23 处位差；改后两层在开启服务 BI 的编译 prefill 和逐 token 执行中均为零。[门控归约源码](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/models/deepseek_v41/engram/common.py)
- **采样。** Ascend 的 `AscendTopKTopPSampler.forward_native` 在 BI 下转入 vLLM 的上层实现。这是该采样入口的规则；训练 loss 使用的已选 token logprob 另有 V2 评分函数。[采样入口](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/sample/sampler.py) · [V2 评分](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/worker/v2/sample/logprob.py)

设备侧已有一组可核查的对照。**历史 Top32** 的 rank 0 在稳定解码步里，普通矩阵乘从原生 `MatMulV3` 变为 `MatMulV3BatchInvariant`，两侧均为 308 次/步。专家矩阵乘仍记录为 `GroupedMatmul`，均为 80 次/步。稀疏注意力从 `MqsmlaKernel___call__` 变为 `SparseFlashMla`，均为 40 次/步，后者对应下一节的 BF16 缓存配方。这些调用次数来自单机 Top32 的采样窗口；全量八机的结论来自真实 loss 和权重同步核验。[逐 rank 设备记录](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/blob/8d91b3356a33e8580259978cf1fc551502faf9e7/experiments/2026-11-19-countbenchqa-performance/evidence/night21-final/decode-profiles.json)

## 4. 让分段训练与增量推理读到相同历史

| 环节 | 原生路径的差异 | 验收版的处理 |
|---|---|---|
| KV 的值与稀疏索引 | A5 滑窗 KV 使用带 scale 的 FP8，压缩 KV 和索引 K 使用打包 FP4。即使历史相同，量化也可能改变数值或 top-k 边界。 | 推理保留分页和增量写入，滑窗 KV、压缩 KV、索引 K 改存 BF16。训练式索引器按页表取 K，将 Q、K 转 FP32 打分，按可见范围和候选块筛选，再取 top-k、按位置排序。decode 图用固定容量和设备端实际长度。 |
| 稀疏注意力与 CP4 | 原生训练 CP4 用本段 Q、跨卡 KV 和显式索引执行 eager attention；最初 verl Actor 的布局校验只接受 TND 或默认值。 | verl 接通 BSND，按压缩块对齐序列，图文融合后切段。每个 rank 把本段 Q、截至本段末尾的连续原始和压缩 KV 前缀送入 SparseFlashMla，用右下对齐滑窗恢复全局坐标，补齐 Q 使用合法物理索引。推理使用已有 CANN 算子的 BF16 分页接口。 |
| 投影、旋转、路由 | 频率表、FP32 旋转、BF16 舍入或专家加权顺序的差异可逐层传播。 | 推理对齐训练侧 RoPE 算序、RMSNorm 后的路由输入、FP32 gate、top-k 归一化、SwiGLU 与共享专家相加顺序。模型级开关与 BI 的全局算子注册分别控制这些路径。 |
| 图文输入与所选 token 评分 | 图像展开、换行或输出词表分母不同，会改变同一回答的分数。 | 核对原图与输入 ID，输入保留图片占位符，输出屏蔽 ID `129264`。Actor 从自身 logits 调推理同款 V2 评分前向，使用解析反向传回梯度。R19 在本 CP 段计算所选 token 分数，再拼回序列，省去汇集完整词表 logits。 |

KV 页、训练式索引器分别由 `VLLM_ASCEND_DSV41_BF16_CACHE` 和 `VLLM_ASCEND_DSV41_TRAINING_INDEXER` 控制。索引器用 PyTorch 基础算子实现选择逻辑，注意力调用 CANN SparseFlashMla。推理按原存储的物理行写回非连续缓存视图，使下一步 decode 能读到新值。[KV 规格](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/models/deepseek_v41/cache_config.py) · [索引器](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/ops/dsv41_training_indexer.py) · [分页 SFA 封装](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532/vllm_ascend/attention/sparse_flash_mla.py) · [训练 CP4](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/214caa2dce72a03363ed84f3f8b41a2bd72dbe06/fsdp_turbo/models/deepseek/deepseek_v41/context_parallel.py) · [Actor 接入](https://gitcode.com/EliasKaslan/dsv41-verl/blob/17fe22595de5d6112e0d3ec8f579f31bf5c2e28a/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py)

还需要验证图回放如何使用这些算子。索引器在设备上读取本次请求的长度和页表，避免捕图时固化旧元数据。mHC 通过预注册算子进入解码图，MoE 固定槽位的物理形状，同时允许每次回放选择不同专家。验收先用固定 token 历史换批、换序和更换页表，再以真实生成、带梯度评分和更新后的下一轮生成覆盖完整训练流程。

## 5. 性能看同一批题、同一计时窗口

全量对照采用相同的 1,024 输入、2,048 输出负载。下表均为第 10–12 步均值，单位为秒。整步还包含阶段协调等工作，单列各阶段的时长无需与整步严格相加。

| 全量配方 | Rollout | Actor 更新 | 权重同步 | 整步 |
|---|---:|---:|---:|---:|
| R15 基线 | 98.13 | 76.73 | 9.94 | 185.31 |
| R181，CP 本地评分，仅训练进程关闭透明大页 | 97.48 | 69.85 | 10.01 | 177.84 |
| R19，基于 R181 关闭全部 40 层激活重计算 | 98.23 | 65.75 | 12.02 | **176.46** |

R19 相对 R15 的整步平均缩短 **4.78%**。R181 与 R19 的直接对照中，训练阶段在三个采样节点上观测到的最高卡显存从 57,814 增至 83,942 MiB。两组均完成 12 步严格验收；R19 的 768 条生成历史及对应分数字节与 R15 的同题窗口相同。按 R15 更长的第 10–28 步窗口计算，平均整步为 186.98 秒，其中 rollout 为 98.16 秒，占 52.50%。另测的 R20 改为按物理页读取 BF16 索引键，两步数值均通过，第二步 rollout 却由 99.06 增至 112.13 秒，因此实长配方沿用 R19 的读法。

**历史 Top32 的性能口径不同。** 同机 96 条固定请求中，原生、优化前对齐、mHC 优化后对齐分别为 **43.747、143.791、113.352 秒/批**。同时加回 BF16 KV 页和训练式索引器增加 35.11 秒；加回全局 BI 及其自动选路增加 22.00 秒。两组消融覆盖的执行路径有重叠。在线 RL 暖步另用可反传的 TND 兼容对照，分别为 239.39、333.60、274.80 秒/步。[Top32 联合消融及对照](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/blob/8d91b3356a33e8580259978cf1fc551502faf9e7/experiments/2026-11-19-countbenchqa-performance/evidence/night21-final/triplets.json)

## 6. 对应哪一版代码

四仓统一交付分支为 `experiment/countbenchqa-1125`。下表按提交固定源码。复现上午的实机结果还需匹配实测镜像、权重配置、启动配方与当次源码覆盖包；分支头记录的是已提交代码。历史 Top32 百步配方保留在四仓的 `optimization/countbenchqa-zero-20261119` 分支。

| 仓库 | 1125 提交 | 历史 Top32 提交 |
|---|---|---|
| vLLM Ascend，推理与 BI | [`2dbf927dd`](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/tree/2dbf927dd7a27f8aaa8819b1c0bab3e2662ce532) | `aa9500af` |
| FSDPTurbo，模型与反向 | [`214caa2`](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/tree/214caa2dce72a03363ed84f3f8b41a2bd72dbe06) | `cdd0c570` |
| verl，Actor 评分与权重同步 | [`17fe2259`](https://gitcode.com/EliasKaslan/dsv41-verl/tree/17fe22595de5d6112e0d3ec8f579f31bf5c2e28a) | `798cfee1` |
| 实验与验收 | [`8d91b33`](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/tree/8d91b3356a33e8580259978cf1fc551502faf9e7) | `8bd92c42` |

复核全量结果时，按运行号检索归档 `full384-cp4-r15`、`full384-cp4-r19`。每包内的 `resolved_launch.yaml` 是实际解析配置，`workers/` 记录推理进程及加载源码，`audit/step*.safetensors` 保存参与 loss 的原始张量。R19 使用 `DSV41_FIXED_LENGTH_1K2K=1`、`DSV41_TRAIN_STEPS=12`、`DSV41_CP_LOCAL_SCORES=1`、`DSV41_DISABLE_ACTOR_THP=1`、`DSV41_ACTOR_RECOMPUTE=0`。实测覆盖包 `全量主机内存与重计算实验.overlay.tgz` 的 SHA-256 为 `8ab24e7cda4619acdd074163eebd2b8dd7544f0e80a11bdfc04be387f24bf055`。
