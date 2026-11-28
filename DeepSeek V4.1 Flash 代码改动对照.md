# DeepSeek V4.1 Flash｜原生基线与最终版代码改动对照

**截至 2026-11-27，40 层、每层 384 个路由专家，8 机 64 卡。** 验收比较的是同一版权重、同一图文前缀、同一回答 token 在两处的 FP32 logprob：一处是 vLLM 生成时记录的分数，另一处是 Actor 用自身 logits 算出、实际进入训练 loss 的分数。自然 EOS 长训完成 85 个更新步，5,440 条回答、122,341 个有效位置全部逐位相同。

**两端采用了不同的对齐方向。** 模型内部的 RoPE、mHC、专家合并、KV 精度和稀疏选择主要由**推理向训练**看齐；最后一步从 logits 取已选 token 分数，则由**训练向推理**看齐。CP4 的重点是多卡分段后仍等价于整段训练；BI 关注同一请求换批次，PD 关注同一历史在 prefill（整段预填充）和 decode（逐词解码）时的结果。性能、容量、梯度正确性和观测改动在表中单列目的。

| 对照范围 | 本文采用的口径 |
|---|---|
| 原生源码 | 以附录中的 Git 提交为起点，比较端点代码。包括项目收到的模型运行前置适配。 |
| 原生实机对照 | 原生推理插件、原生数值开关；为跑通同一全量权重、CP4、在线更新和测量，两组共用部分训练入口、加载及资源管理代码。源码新增的能力可能已经用于两个实测组。 |
| 最终版 | 全专家 BF16；训练 TP1 / FSDP64 / EP64 / CP4，推理 TP1 / DP64 / EP64 / PCP1。运行效果以归档启动实参、源码挂载及有效 token 复核为准。 |

## 一、训练侧｜FSDPTurbo 与 verl Actor

### 1. 构造同一个模型，喂入同一段图文

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| 模型构造与检查点映射 | 通用加载入口和早期裁剪配方无法单靠层数还原完整视觉、文本、专家和 Engram 配置 | 从全量检查点映射模型字段与权重名，恢复 40 层、384 路专家和第 1、14 层 Engram；冻结视觉塔和两张 Engram 主存表 | 模型完整性；两端共同对齐检查点 |
| 专家权重实例化 | 全量 routed expert 的 gate/up、down 参数可在每个进程先完整分配，再切本地分片，推高主存峰值 | 支持在 meta 设备上声明形状及 dtype，确定 EP 归属后直接加载本 rank 切片；前向前完成实体化 | 容量与启动；无单向数值对齐 |
| 图文样本与微批 | 数据集、rollout 和 Actor 分别处理图像、模板与 token；重新加工图像、漏掉换行或错配图像跨度都会改变前缀 | 保留原图与展开后的输入 ID，逐样本关联图像范围、token 类型与回答 ID；微批切分和恢复时使用相同的 tokenizer/processor 约定 | 输入一致；训推双方对齐同一原始样本 |
| 冻结视觉的注意力 | 训练 NPU 视觉融合路径使用 FlashAttentionScore，参考 SDPA 与推理 BI 视觉注意力也可能给出不同舍入 | 为冻结视觉提供 BI FIA 的 TND 前向，保留单图完整双向注意力；另有 CPU SDPA 参考。该 BI 路径只处理不需要视觉梯度的输入 | 视觉注意力：训练向推理；CPU 实现用于功能校验 |
| mHC 的训练前反向 | 推理 HcPre/HcPost 与训练 delayed-premix 算序不同；在训练端使用融合前处理还需要保存反向依赖 | 训练侧提供可复用的确定性前处理与分块，保存 pre/post 中间量供反向；推理复用这套语义，具体小批优化见第二部分 | 梯度正确性；模型算术由推理向训练 |

#### 冻结视觉与 mHC 的前向、反向边界

| 算子 | 原生调用 | 最终调用与数据流 | 目的与方向 |
|---|---|---|---|
| 视觉注意力 | NPU 融合入口 `torch_npu.npu_fusion_attention` 对应 FlashAttentionScore，可用其反向 | 冻结视觉 BI 分支把单图 BF16 的 Q/K/V 从 [1,S,N,D] 整理为连续 TND，以真实序列终点 S 调用 `npu_fused_infer_attention_score_batch_invariant`，再恢复批次维。图像内保留完整双向注意力。只接受无需梯度的 Q/K/V | 训练视觉 attention 向推理 BI FIA 对齐；此分支的适用条件是冻结视觉 |
| mHC 前处理 | 训练模型采用 delayed-premix 语义；推理原生 HcPre 采用自己的归约路径 | 训练封装调用 `mhc_pre_sinkhorn_premix`，保存反向需要的状态；反向用 `mhc_pre_sinkhorn_premix_backward`，将梯度传回输入、混合参数及 premix。推理侧只复用前向算术 | 推理向训练；训练保有可微前向 |
| mHC 后处理 | 各端分别完成子层输出与多路残差的合并 | 训练调 `mhc_post`，反向调 `mhc_post_backward`；推理使用同款 post 前向。训练 BI 前、后处理均以 128 行为物理块，尾块补零，只保留有效输出 | 残差混合由推理向训练对齐；128 是一次调用的 token 行数 |

源码入口：[检查点与 Actor](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py) · [训练侧模型适配](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/models/deepseek/deepseek_v41/adapter.py) · [冻结视觉](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/ops/npu/vision_flash_attention.py) · [mHC](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/ops/npu/mhc.py)

### 2. CP4｜分四段计算，保留每个 token 的完整可见历史

TND 中 T 是打包后的总 token 数，N 是头数，D 是头维；BSND 显式保留批次 B 和序列 S。底层原生训练已具备 CP 与 KV-AllGather；早期项目的 verl Actor 入口对布局有额外限制。下表分别说明入口、通信和注意力算子的变化。

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| Actor 入口与切分 | 早期 Actor 按 TND 打包输入，完整模型的图文融合、BSND 与 CP 分段未贯通 | 先构造完整的图文 embedding，再按压缩块与 CP 边界补齐、切四段。保留原始有效长度、全局起点及回答掩码 | CP 接入；分段训练向整段训练对齐 |
| 稀疏注意力前向 | 原生 CP 在每段计算本地 Q，AllGather 原始 KV；有压缩时把原始与压缩 KV 拼接，滑窗索引与压缩 top-k 也拼接，再调 `indexed_sparse_attention` | 保留本段 Q 与 KV-AllGather。BSND 分支把**到本段末尾的原始 KV 前缀、压缩 KV 前缀、压缩索引和滑窗参数**分别交给 SparseFlashMla。右下因果对齐使本地第一个 Q 仍能读取上一段窗口内的历史 | CP 语义与训推注意力对齐；训练改用与推理对应的 SFA 算子路径，训练的可见范围仍以全序列定义 |
| top-k、补齐与跨层状态 | 物理补齐若参与 top-k，会改变候选数；若给补齐 Q 留越界索引，即使不计 loss，算子仍可能读取它 | 有效 Q 按真实长度选择并排序；补齐 Q 提供合法索引；保留压缩 KV、索引键和候选块的生产层与消费层关系 | 边界正确性；分段训练向有效序列对齐 |
| 张量布局与反向 | AllGather 切片或压缩器视图可带非连续步幅；仅让前向算对，无法保证梯度回到正确输入 | 入融合算子前整理 Q、原始及压缩 KV 的布局；反向按实际长度、压缩信息调用 `sparse_flash_mla_grad`，回传 Q、KV、sink 梯度。整层重算时恢复本次前向的跨层状态 | 训练算子与梯度正确性 |
| TND 兼容 | 单文档长度不能整除压缩比例时，旧入口会按对齐要求拒绝输入 | 单文档保留全部原始 token，仅舍弃不能构成完整组的压缩尾部；多文档打包仍检查内部边界 | 兼容原 TND 输入；独立于最终 BSND 配方 |
| 静态批量检查 | 设备总数包含共同处理一条样本的 CP rank，按设备数检查全局批量可能误拒绝 12 题 × 8 回复 | verl 按实际承载不同样本的数据副本数检查整除性，让同一批量可用于两组实机对照 | 运行正确性；共同支撑改动 |
| 反向设备识别 | SparseFlashMlaGrad 的设备型号查询接口与当前 NPU 后端不匹配 | 经 torch_npu 后端获取设备名，按对应设备运行梯度算子 | 运行兼容；训练侧 |

#### 同一条 CP4 序列，注意力实际怎样算

**原生 CP4。** 本 rank 持有本段 Q，`kv_allgather` 沿序列维汇集 KV。原生 CP 调用点没有传 `use_sparse_flash_attn=True`；NPU 注册入口因此进入 `indexed_sparse_attention_torch`。它先 `gather` 选中的 KV，用第一次 `einsum` 算 Q·K，转 FP32 并缩放，将无效索引遮为负无穷；再把 sink 分数接入 softmax 分母，取出真实 KV 的概率、转回 KV dtype，用第二次 `einsum` 加权得到输出。整个过程可在 NPU 张量上执行。[原生 CP 调用](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/26542a18060d72a7346571ed69da980938023578/fsdp_turbo/models/deepseek/deepseek_v41/context_parallel.py) · [原生后端分派](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/26542a18060d72a7346571ed69da980938023578/fsdp_turbo/ops/npu/sparse_attention.py) · [逐项计算实现](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/26542a18060d72a7346571ed69da980938023578/fsdp_turbo/ops/cpu/sparse_attention.py)

**最终 CP4。** 保留同一 `indexed_sparse_attention` 公共入口，在 BSND 分支显式选择已有的 CANN SparseFlashMla。修改集中在如何给算子提供等价的全局历史，以及如何让反向使用同一组输入。

| 算子阶段 | 送入什么 | 算子产出及边界 |
|---|---|---|
| KV-AllGather 和前缀切片 | 本段原始 KV、压缩层产生的 KV；本段 Q 留在本 rank | 原始前缀止于本段末尾，压缩前缀按压缩率截取；显式复制成稠密连续存储，消除集合通信缓冲区切片留下的物理步幅 |
| `sparse_flash_mla_metadata`、`sparse_flash_mla` | 本段 Q、独立的 ori_kv / cmp_kv、压缩 top-k、FP32 sink 与 scale；原始滑窗设置左宽 W−1、右宽 0 | 得到本段 attention 输出及反向所需的 softmax LSE（对数归一化因子）。例如本段从全局位置 128 开始，若窗口 W=128，则本段第一个 Q 可读原始 KV 的位置 1–128 |
| `sparse_flash_mla_grad_metadata`、`sparse_flash_mla_grad` | 上游梯度、前向结果、LSE、Q、原始/压缩 KV、压缩索引、sink，沿用前向的窗口和压缩率 | 梯度回到 Q、原始 KV、有压缩时的压缩 KV 及 sink；离散的 top-k 位置只指示读取位置 |

这个改动同时满足两个目标：分段后每个 Q 的可见历史与整段计算一致，训练还能使用与推理相对应的融合注意力前向。CANN 前反向接口在原生后端已存在，项目接通了 CP4 的输入布局、索引、前缀和梯度路径。

源码入口：[CP 前缀、索引和算子调用](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/models/deepseek/deepseek_v41/context_parallel.py) · [稀疏注意力封装](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/ops/npu/sparse_attention.py) · [批量校验](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/utils/config.py)

### 3. 专家、最终评分与真实 loss

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| 空专家分片 | 原实现根据本 rank 的专家接收量决定整条排列与合并是否切 eager。本地接收 0 项时，发给别处的 token 仍可能返回，此时最后一次合并有真实数据 | 分派、回收均沿用 **AllToAll**。每次 permute/unpermute 按自己的索引判空；空索引走 eager，非空的源端合并保留融合算子，以 FP32 权重累加后转 BF16 | 训练修复空分片的精度分支，恢复两端共同采用的训练式合并 |
| Actor 已选 token 的 logprob | 各框架可用不同的 log-softmax 与取值实现；相同 logits 也可能产生 FP32 末位差 | Actor 用**自己的 logits**调用 vLLM V2 的 `compute_token_logprobs` 前向。解析反向用 FP32 计算 `one_hot(label) − softmax(logits)`，把上游梯度传回 Actor | 最终评分：**训练向推理**；附加训练可用的反向 |
| 评分温度与掩码 | 微批字段可能包装成标量而非 Tensor；单位温度若被误判，已选 token 会回落到另一评分实现 | 解包温度并兼容标量和张量；在满足温度、标签和布局条件时进入对应评分分支，维持同一有效词表 | 评分入口：训练向推理 |
| CP 本地评分 | 先汇集各段的完整词表 logits，再统一取已选 token，跨卡搬运量随词表大小增长 | 在本段 logits 上按全序列的下一 token 标签评分，处理跨段的一位偏移；只汇集逐 token 分数，保留梯度与变长输出。无需辅助输出时可省重复评分 | 通信与显存优化；保持上一行的评分算术 |
| loss 与一致性验收 | bypass 下旧策略字段可引用 rollout 分数；拿这两个字段比较会产生假零 | 在 worker 初始化前设置 bypass；直接从真实带梯度 loss 入口取 Actor 分数，按回答掩码比较 FP32 位。CP 代表 rank 写出记录，跨数据副本统计有效位置 | 验收可信度；比较两端独立前向 |
| 屏蔽词与 GRPO | 屏蔽词的零概率参与 entropy 可产生非有限值；组内相同的浮点奖励也可能因方差计算留下残差 | 屏蔽词按零熵贡献处理；同题奖励先中心化再计算组内方差。计数奖励支持长整数字符串，逐逻辑请求设种子以便同题重放 | RL 数值稳定与复现；不改变模型对齐方向 |
| FSDP 梯度与重算 | 包装输出时需让 FSDP 找到 logits；整层激活重算会重复模型前向 | 输出可按 PyTree 遍历，保留反向依赖；在显存允许的配方中关闭整层重算并实测核对。85 步启动实参为 `recompute: false` | 训练正确性与性能；以激活显存换计算 |

**最终评分的算子边界。** 原训练评分可以通过交叉熵或 log-softmax 取值；两端即使数学等价，也可能在舍入或零的符号上不同。最终在 NPU 上将标签整理成连续的 INT64 [有效行数,1]，把 Actor 自身的连续 logits 和标签送入 vLLM V2 的 `compute_token_logprobs`，每行只返回被采样 token 的一个分数。训练新增的一阶反向用 FP32 softmax 算整行概率，先乘负的上游梯度，再通过 `gather/scatter_` 给目标词加回上游梯度，最后转回 logits dtype。CP 本地评分在切段前先确定全序列下一 token 标签，跨卡 `all_gather_tensor_autograd` 汇集的是分数；这样段末 token 仍对应下一段的正确标签。

以空分片为例，在双卡同输入对照中，一侧收到 232 项，另一侧收到 0 项。旧实现有 5,026 / 14,848 个 BF16 输出字不同；修复后，输出、输入梯度和路由权重梯度均逐位相同。这说明需要保住的是**返回后有数据的加权求和**。[双卡前反向原始结果](空专家分片双卡HCCL逐位及反向实测.json)

源码入口：[专家分派与合并](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/distributed/expert_parallel/dispatcher.py) · [已选 token 评分与梯度](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/utils/dsv41_logprob.py) · [loss 入口](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/workers/utils/losses.py) · [验收器](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/utils/debug/train_rollout_consistency.py)

### 4. 训练更新、资源和权重交接

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| 分片导出与精度 | 聚齐全套专家再切片占用主存；统一按 BF16 导出会丢失检查点中敏感参数的 FP32 位 | 按 EP 归属导出本地专家，按参数归属与 dtype 组织分片；mHC、router、sink 等敏感参数保留原精度。接收端走正式权重加载路径 | 更新后策略一致；**推理接收并对齐训练端新权重** |
| 多机放置与传输 | 单机映射不足以区分多节点上的本地 rank、全局 rank 和副本；不同通信角色也需独立端口 | 明确节点、DP/EP 归属及传输组，区分训练与推理端口；更新与同步分别记时，按策略版本核对发送、接收哈希 | 分布式正确性；训练 → 推理 |
| Host 内存与训练吞吐 | 基线沿用进程主存页策略；不同大小的首次梯度卸载共享锁页策略 | 支持仅对 Actor 关闭透明大页、限定首次梯度锁页大小。长训实际启用 Actor 侧关闭透明大页；梯度阈值属于可选实验 | 性能与容量；各选项须按实参归因 |
| 可复现恢复与定点测量 | 仅有权重或总耗时不足以重建图文输入、区分审计开销与实际更新 | 随检查点保存 processor 元信息；支持实际 worker 的 CPU 亲和性配置、定点 NPU 采集与审计计时，分别记录生成、更新及权重同步 | 恢复正确性与性能诊断 |
| 失败定位 | 单个汇总 logpdiff 无法区分输入、策略版本、层输出或最终评分差异 | 保留失败位置的 token、图像与权重指纹；按相同轨迹、CP 分片和层配对前向，在分布式错误传播前落盘；优化器分片抽样核对更新前后状态 | 诊断与验收；无单向数值对齐 |

源码入口：[Actor 加载与导出](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py) · [训练参数卸载](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/blob/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e/fsdp_turbo/distributed/fine_grained_fully_shard/param_group.py) · [权重接收审计](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/workers/rollout/vllm_rollout/dsv41_weight_precision_audit.py) · [实验与诊断代码](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/tree/84d8f57dbf54b2ac60b375dad42edc743c478b20)

## 二、推理侧｜vLLM Ascend 与 rollout worker

### 1. BI 与模型算术｜同一请求换一个批次，仍执行可复现的计算

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| BI 分派与确定性 | 上游已有 BI 注册机制，原生实机组关闭 BI、观测到等级 0。初始化也可能覆盖进程已设定的确定性等级 | 验收组开启 BI，在 worker 内设置并读回 NPU 等级 3；保留通信确定性和适配的权重布局。两组推理都使用异步调度与 decode 图 | BI、可复现性；推理端跨批次对齐 |
| 矩阵乘与 softmax | 常规后端按形状与可用实现选择计算路径 | AscendC 可用时优先接管 mm/matmul；仅 Triton 可用时可接管 mm/matmul/linear；Triton 可用时另注册 addmm、bmm、softmax 和 _softmax 的 BI 版本。专家矩阵乘仍走 GroupedMatmul | 启用已有 BI 能力；推理内部对齐 |
| 归一化、注意力与求和 | 常用融合 AddRMSNorm、通用注意力及原生求和；调度器可显式指定求和输出缓冲 | AscendC BI 将融合残差归一化拆为加法与 RMSNorm，并注册 BI 融合推理注意力；模型的稀疏注意力见下一节。满足条件的尾轴 sum 走 BI，显式 out 等调用回退原生并写回指定缓冲 | BI 与调用语义；推理内部对齐 |
| mHC 残差混合 | 推理原用 HcPre/HcPost，其算序与训练 delayed-premix 不同 | 前处理采用训练同款 mhc_pre_sinkhorn_premix，后处理采用 mhc_post。对符合 A5、确定性 3、BF16、隐藏维度 5,120、四路 mHC 等条件的调用，1–15 行补至 16，16–3,072 行按实际行数运行；其余走既有分块 | 算术：推理向训练；最小 16 行是减少小批填充的性能优化 |
| 图文视觉与输出词表 | 推理视觉线性层、图片占位展开及输出词表处理与 Actor 存在差异；将图片占位符计入输出分母会影响普通 token 分数 | 视觉线性计算采用训练对应算术；核对图文展开与换行；输入保留图片占位符，输出屏蔽 ID 129264 | 视觉线性与输出词表：推理向训练；输入由两端共同对齐 |
| Engram 门控 | 原融合门控点积及全局 BI 对 Tensor.sum 的接管都可能改变归约顺序 | 按训练次序计算门控，显式调 ATen 尾轴求和。真实权重下，第 1、14 层原先各有 4、23 个 BF16 元素不同，改后受测 prefill 与逐词执行均为 0 | Engram 算术：推理向训练；同时保持 PD 一致 |

#### BI 注册与模型专用算子

BI 有两层作用。通用注册决定哪些 PyTorch 入口可以替换；模型代码还决定一次具体调用是否绕过通用入口。下表列出最终代码中需要区分的路径。

| 原生入口 | 对齐版算子或调用顺序 | 生效条件与目的 |
|---|---|---|
| `aten::mm / matmul / linear` | AscendC 扩展将前两者注册为 `npu_mm_batch_invariant / npu_matmul_batch_invariant`；只有 Triton 后端时，三者分别注册为 Triton BI 实现 | 推理跨批次 BI。单机 Top32 设备记录观察到普通矩阵乘由 MatMulV3 切至 MatMulV3BatchInvariant；专家分组矩阵乘仍是 GroupedMatmul |
| `aten::addmm / bmm / softmax / _softmax` | Triton 可用时注册相应 BI 入口；softmax 与 _softmax 使用同一个 BI 实现 | 推理跨批次 BI。注册是后端能力，某次模型前向是否调用要看实际路径 |
| `npu_add_rms_norm` 与通用 FIA | 将融合残差归一化拆成残差相加再 RMSNorm；将 `npu_fused_infer_attention_score` 指向 BI FIA | AscendC BI 生效时使用；冻结视觉 attention 和文本稀疏 attention 分别见两侧专节 |
| `torch.sum / Tensor.sum` | 单个尾轴、支持 dtype 的 NPU 输入且无显式输出 dtype 时可调 `npu_reduce_sum_batch_invariant`；传入 out 缓冲区时走原生写回 | 推理跨批次 BI，同时保留调度元数据的原位写回语义 |
| 视觉投影 `F.linear` | patch 投影、QKV、输出投影、视觉 MLP 和投影到文本空间的线性层，在训练式视觉开关与 NPU 条件下显式调 `torch_npu.npu_linear` | **推理视觉线性向训练对齐**；方向与训练侧冻结视觉 attention 的适配相反 |
| Engram 的门控归约 | 对齐路径在 FP32 中计算 hidden/key 的均方倒根及逐通道乘积，再用 `aten.sum.dim_IntList` 沿末轴累加；经缩放、绝对值下限、平方根、符号选择、sigmoid 得到门值，最后屏蔽无效 token | **推理向训练**。显式 ATen 入口保留训练的门控归约；通用 BI 对 Python `Tensor.sum` 的替换仍可服务其他调用 |

**mHC 的 16 行和 128 行怎样共存。** 原生 HcPre 可能随 token 行数改变隐藏维 K 上的归约划分。适配后，推理改用训练同款 premix 前向；该 5,120 维、确定性等级 3 的实现固定 K 的划分。小批优化只减少为独立 token 补出的行。下表以一次调用收到的行数为例，推理列要求该前处理或后处理分别命中设备、精度、形状和开关条件。

| 有效行 | 训练 BI 前/后处理 | 推理命中优化时的前/后处理 |
|---:|---|---|
| 4 | 补到 128，取前 4 行 | 补到 16，取前 4 行 |
| 32 | 补到 128，取前 32 行 | 一次算 32 行 |
| 129 | 两次各算 128 行，第二次只取 1 行 | 一次算 129 行 |

这里的性能收益来自减少填充计算。最终模型的 85 步逐位核验覆盖实际运行的输入；其他设备、形状或确定性等级仍由源码条件决定是否回到 128 行分块。

源码入口：[BI 注册与回退](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/batch_invariant.py) · [mHC 分块](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/batch_invariant_hc.py) · [视觉](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/models/deepseek_v41/vision.py) · [Engram 门控](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/models/deepseek_v41/engram/common.py)

### 2. 稀疏注意力｜prefill 和 decode 读取相同位置、相同数值

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| RoPE | 原生推理旋转实现与训练的频率、取位及中间舍入路径不同 | 用与训练一致的频率表和全局位置，按训练次序 FP32 旋转后转回 BF16。增量步以本 token 的全局位置索引 | 推理向训练；同时对齐 PD 位置语义 |
| KV 页格式 | A5 原生滑窗 KV 为带 scale 的 FP8；压缩 KV 和索引 K 为打包 FP4，读出后已经历有损量化 | 滑窗 KV、压缩 KV、索引 K 均进入 BF16 分页缓存；保留页表和增量写入机制。SFA 读取选中的 BF16 历史行 | 缓存数值：推理向训练；同时对齐 PD |
| 稀疏索引器 | A5 原生量化索引器以量化键打分，小差异可能改变入选的历史位置 | 项目组合现有基础算子：BF16 Q/K 转 FP32 做点积、ReLU、逐头加权求和；按真实可见范围屏蔽，按层生成或消费候选块，选 top-k 并按位置排序交给注意力 | 选择规则：推理向训练；属于新实现的索引逻辑 |
| decode 图回放 | 不同请求的页表、当前位置与 KV 长度会变化；把捕图时的长度固化会读到无效历史 | 固定候选容量，把长度、位置、物理页号作为设备侧运行时输入；填充请求用安全页，屏蔽无效键和超出因果范围的位置 | 图正确性、PD 对齐；推理内部 |
| KV 写回 | BF16 页可有步幅或存储偏移；单行 decode、大页池场景中旧写槽可能落在目标页第 0 行 | 保留底层存储别名，按真实步幅与偏移定位写槽；单行场景把物理页号和页内偏移换成绝对行号，复用设备 PA 行写。无效槽保持跳过 | PD 正确性；保证下一步读到刚写入的 KV |
| 分页读键 | 较早的对齐版已能读对 BF16 键，但二维高级索引可触发整片 Slice 和 AI CPU Index | 满足列连续、单头等布局约束时，将原存储按真实页步幅映射成二维物理行，单轴 index_select 按原顺序取键；其余布局回退。保持读取值、顺序、打分与 top-k 一致 | 性能优化；维持已经验收的推理向训练对齐 |
| 稀疏注意力调用 | 原生推理消费量化缓存和原生索引。单改缓存 dtype 不能确保读到同一组历史位置 | 将已对齐的页、稀疏位置、滑窗及长度元数据交给已有 CANN SparseFlashMla；prefill 与 decode 各保留自己的输入组织 | PD 与训推对齐；推理历史数据向训练靠齐 |

#### 从一次 decode 看索引、写槽与注意力的分工

| 步骤 | 原生或优化前的算子 | 最终执行的算子与数据流 | 目的与方向 |
|---|---|---|---|
| ① 写入当前 KV | SK 行写以二维坐标定位；大页池上的 BF16 单行写曾落到目标页第 0 行 | 对命中的连续缓存，计算 slot = 物理页号 × 每页行数 + 页内偏移，以 `DeviceOperator.scatter_cache` 走 PA 行写；负槽跳过。非连续页另按真实存储步幅和偏移写回 | PD 正确性；确保下个 decode 和整段重放读取同一行 |
| ② 读取 BF16 索引键 | 早期训练式索引器用页号、页内偏移做二维高级索引 | 符合布局条件时，以 `as_strided` 建同一底层存储的行视图，再以 `index_select` 单轴取键。物理行号按实际页间步幅计算，保留页间空隙。此规则与上一步连续缓存的寻址条件不同 | 性能；读取值与候选顺序保持一致 |
| ③ 给可见历史打分 | 原生 A5 使用量化 Lightning Indexer 系列接口，以打包索引键计算候选 | 新增训练式索引逻辑；图内以 FP32 `einsum` 得到每头点积，依次执行 ReLU、乘头权重、跨头 sum。用设备侧实际长度和全局位置屏蔽无效项为负无穷 | 推理向训练，兼顾 PD；这里输出分数，还未执行 attention |
| ④ 选择压缩位置 | 原生量化索引器返回选中位置与候选信息 | 在候选生成层按块做 amax / topk，并保留最新可见块；在候选消费层按块 ID 屏蔽，再做 token topk、按位置排序，将无效项置 −1 并补到固定宽度 | 推理向训练；固定宽度支持图回放，实际可见范围随请求变化 |
| ⑤ 计算注意力 | 原生走面向其缓存格式的稀疏注意力入口 | BF16 分支把 Q、原始滑窗页、压缩页、页表和选中位置交给 `sparse_flash_mla`。索引器负责“选哪些历史位置”，SFA 负责“这些位置对当前 Q 的贡献” | 推理向训练的历史精度与选择规则对齐；prefill/decode 共用其语义 |

第③④步是项目用现成基础算子写的完整索引逻辑；第⑤步复用已有融合算子。分页读键的优化只改第②步。Top32 设备跟踪中，旧读法连续六次 decode 触发的 48 次整片 Slice 和 48 次 AI CPU Index，在单轴取键后均降为 0。该结果说明了取键的收益来源。[读键性能对照](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/blob/bb70823988001ee45fe05f429de3f4ed80b783e5/experiments/2026-11-25-top32-singlehost/release/performance.json)

源码入口：[页格式](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/models/deepseek_v41/cache_config.py) · [RoPE](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/rope_dsv4.py) · [索引器及读键](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/dsv41_training_indexer.py) · [写槽](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/dsv41_cache_store.py) · [分页注意力](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/attention/dsa_v41.py)

### 3. MoE｜通信负责搬运，专家加权保持训练顺序

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| 路由与专家计算 | 推理的 router 输入舍入、门控精度、top-k 归一化、SwiGLU 截断及共享专家加权与训练路径可有差异 | 对齐 RMSNorm 后转 BF16 的位置、FP32 gate、top-k 权重与专家激活；统一共享专家贡献的计算次序 | 模型算术：推理向训练 |
| EP 分派与回收 | 推理可随批大小切换分派策略；先在各卡加权再跨卡浮点求和会改变舍入位置 | 对齐配方固定 AllGather 分派。每个 token 选出的 6 个专家各占一槽，只有所属 rank 提供该槽未加权的 BF16 输出，其余填零。按整数位模式 ReduceScatter 归并，再按原 top-k 槽顺序用训练式 unpermute 做一次 FP32 加权，最后转 BF16 | 分派固定用于 BI；浮点合并顺序由推理向训练对齐 |
| 图回放层状态 | 重放部分解码层时若丢失当前 MoE 层游标，会引用错误层的路由状态；复用槽位还可能带入旧值 | 重放时继承层游标，在槽缓冲区内原地清零缺失槽，避免再分配完整 DP 缓冲 | 图回放正确性；推理内部 |

#### 一个 token 的六路专家输出如何合并

| 操作 | 原生或原有风险 | 最终算术与算子 | 目的与方向 |
|---|---|---|---|
| 选专家、取权重 | 原生融合门控可经 `moe_gating_top_k_hash`；偏置、选取和归一化的位置都可能影响结果 | 对满足条件的常规路由分支，以训练式稳定 sqrt-softplus 算基础分；用加偏置的分数选 top-k ID，再从**未加偏置**的基础分 gather 权重，按需要用 ATen 尾轴 sum 归一化并乘缩放系数。哈希路由有独立条件 | 推理路由向训练对齐 |
| 专家内部计算 | 原生的激活精度与截断顺序可能不同 | 保留专家分组矩阵乘。对命中的 dense/shared SwiGLU 路径，以 FP32 分开 gate 和 up，按训练规则截断，执行 SwiGLU 后回到输入 dtype | 推理专家算术向训练对齐 |
| 还原槽并跨卡传输 | 各 rank 若先加权再归约，就在不同位置发生浮点舍入 | 按 InitRoutingV2 给出的行映射用 index_select 还原 token × top-k 槽，缺失槽清零。隐藏维 5,120 为偶数，可将相邻两个 BF16 按位视为一个 INT32，再做整数 ReduceScatter；每槽只有一个提供方 | 通信保持位模式；分派固定用于 BI |
| 一次加权合并 | 改动加权顺序或过早转 BF16 会改变结果；训练空分片曾触发不同合并实现 | 恢复 BF16 槽，令槽值、权重转 FP32，按原槽序调用 `npu_moe_token_unpermute`，结果再转 BF16。训练侧 AllToAll 返回后，非空合并同样走融合路径 | 最后一次加权由推理向训练对齐；训练空分片修复保住这条路径 |

[路由分支](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/fused_moe/router/fused_topk_router.py) · [SwiGLU](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/activation.py) · [专家槽收集与合并](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/dsv41_graph_moe.py)

**训练仍用 AllToAll。** 两端需要相同的是送到最终加权算子的专家结果、槽次序、路由权重及累加精度。推理侧整数归并的每个槽只有一个提供方，因此可以保留 BF16 位模式，包括负零。[推理分派](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ascend_forward_context.py) · [固定槽与 FP32 合并](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/ops/dsv41_graph_moe.py)

### 4. 图执行、休眠与权重接收

| 改动点 | 原生行为或原有限制 | 最终实现 | 目的与方向 |
|---|---|---|---|
| 图模式与采样 | 原生实机组已可异步调度并运行 decode 图；不同 worker 的预热完成时间差异大 | 保持 FULL_DECODE_ONLY；核验图档位及数值开关，模型装载后增加全 rank 就绪屏障；两组都使用该屏障。BI 配方中采样的 top-k/top-p 按相应 vLLM 路径执行 | 原生与最终组公平对照、稳定启动；推理内部 |
| 休眠与主机备份 | 推理与训练分时使用 HBM；原分配器的主机备份策略影响可运行性和耗时 | 支持选择普通或锁页主机备份；本次原生对照同样挂载普通内存备份适配，不能把休眠差距归于单项代码。休眠计时取 trainer 调用前后墙钟，包含卸载及等待 | 容量与性能；无数值对齐方向 |
| 同步后再生成 | 只看启动时权重相同，发现不了后续版本的分片错配或敏感权重截断 | 通过正式加载接口同步，按 worker 与版本核对敏感参数哈希；唤醒后用新策略生成，再对照本轮 Actor | 更新后策略一致；推理向训练的新权重对齐 |

源码与运行证据：[主机备份](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/blob/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3/vllm_ascend/device_allocator/camem.py) · [权重接收](https://gitcode.com/EliasKaslan/dsv41-verl/blob/48b8a2a25d9deb41ac072d5282be01b8c9fedaea/verl/workers/rollout/vllm_rollout/dsv41_weight_precision_audit.py) · [两组图模式与数值开关实测](两版图与数值开关实测.json) · [原生与最终配方构造](make_profiles.py)

## 三、验收结果与适用范围

| 验证负载 | 结果 | 它验证了什么 |
|---|---|---|
| 全专家自然 EOS，16 题 × 4 回复 | 85 个完整更新步，5,440 条回复，122,341 个有效 token，FP32 位差 0 | 同一策略版本内，在线 rollout 与带梯度 Actor 前向逐位相等。机时结束时第 86 步未计入 |
| 全专家自然 EOS，12 题 × 8 回复 | 4 个完整更新步，7,813 个有效 token，位差 0 | 修复前第 3 步曾出现 6 个位差；改后在该批量下通过 |
| 全专家固定实长 1K 输入、2K 输出 | 较早长序列配方 28 步 / 3,670,016 个有效 token；优化配方 12 步 / 1,572,864 个，均为零位差 | 长序列覆盖和训练优化有独立验收记录，步数与自然 EOS 分开计 |
| Top32 单机固定历史读键对照 | 同一批历史，整批推理 127.20 → 92.28 秒 | 单独验证物理行取键的收益；不能用作 384 专家在线 RL 的整步加速比 |

全专家 16×4 实测在共同暖步 2–4 的整步平均为原生 417.50 秒、最终 113.67 秒；其中推理休眠为 340.36 和 33.87 秒，扣除后生成与调度为 2.23 和 3.11 秒。整步变化主要落在等待休眠的窗口，现有记录不足以把它归因于某个算子或单纯的权重拷贝。[全专家实测与原故障配对](性能与精度/空专家修复后的全专家RL.md) · [85 步原始张量复核](实测日志/full384-n27-final-long2/原始张量独立复核.json) · [85 步实际启动参数](实测日志/full384-n27-final-long2/启动脚本和实参.zip)

**独立候选的边界。** 训练端改用原生推理量化 KV 和索引器属于反方向的候选；本次验收使用推理 BF16 页和训练式索引器。其余独立开关与专项结果如下。

| 环节 | 原有做法 | 候选修改 | 目的与方向 |
|---|---|---|---|
| Top32 RoPE | 训练式旋转由多个操作完成 | 可选 Triton 单核按相同中间精度与舍入点完成旋转，限支持布局 | 性能；延续推理向训练对齐 |
| PCP8 / TP8 | 全专家验收采用 PCP1 / TP1 | 独立验证 PCP 的图内通信组、跨 rank 放置与权重传输，以及 TP8 专家合并与输出投影 | 并行扩展与一致性；推理不同并行配置间对齐 |
| 索引键按页搬运 | 正确性路径按请求读取 BF16 键 | 曾试过按物理页分批取键，零位差但实测更慢；最终的单轴物理行读取另见推理侧第 2 节 | 性能探索；保持 PD 及训推一致 |
| 二级休眠 | 全量配方使用一级休眠 | 单独测试二级休眠，在预热阶段退出 | 容量探索；未作为长训结论 |

## 附录｜版本与复核

| 仓库 | 原生源码起点 | 最终源码提交 | 两端树差文件数 |
|---|---|---|---:|
| FSDPTurbo | `26542a1` | [6bbc8de](https://gitcode.com/EliasKaslan/dsv41-fsdp-turbo/tree/6bbc8ded3644618ae30e954db0ac78bc4ff7e13e)，含功能修复 191bd21 | 40 |
| verl | `8f89a4f6` | [48b8a2a2](https://gitcode.com/EliasKaslan/dsv41-verl/tree/48b8a2a25d9deb41ac072d5282be01b8c9fedaea) | 58 |
| vLLM Ascend | `fd4be4dbc` | [a45fc71d](https://gitcode.com/EliasKaslan/dsv41-vllm-ascend/tree/a45fc71d241b22ce1cef6aa38c5ecf000d4534c3) | 47 |
| 实验与验证 | `b708d4a` | [84d8f57](https://gitcode.com/EliasKaslan/dsv41-zero-mismatch-private/tree/84d8f57dbf54b2ac60b375dad42edc743c478b20) | 1,410 |

文件数按两端 Git 树统计，包含测试、说明与历史证据。可在各仓库执行 `git diff --name-status <起点> <最终提交>` 复核。部署使用版本化覆盖包，实际装载文件以清单和节点记录为准。原生对照沿用的共同支撑改动可从原生配方构造脚本复查。[原生八节点源码校验](八节点原生源码校验.json) · [16×4 最终清单](final-empty-ep-v1/manifest.json) · [12×8 最终清单](final-batch-empty-ep-v1/manifest.json) · [八节点空分片修复挂载](八节点空专家修复源码挂载.json)
