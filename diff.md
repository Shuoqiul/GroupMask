# GroupMask ICLR'27 修改计划 — 代码变更说明

**日期**: 2026-09-16
**对应计划**: `groupmask_iclr2027_plan.md`（阶段一/二/三/四/五 的可代码化部分；阶段二论文侧工作与阶段六 kernel 未涉及）

---

## 总览

| 文件 | 类型 | 对应阶段 |
|---|---|---|
| `flashlm/compression/hypernetwork.py` | 修改 | 五（prior 注入） |
| `flashlm/compression/semi_pruning_helper.py` | 修改 | 四（uniform 正则）+ 五（prior 计算） |
| `recipe_train_groupsparsity.py` | 修改 | 一/三/四/五（参数、诊断、日志） |
| `hf_ppl.py` | 修改 | 二（N:M baseline）+ 五（eval 重挂 prior） |
| `test_prior_pooling.py` | 新增 | 五（防错位单元测试） |
| `srun_*.sh` × 9 | 新增 | 一/二/三/四/五（实验提交脚本） |

验证状态：所有 Python 文件通过 `py_compile`；单元测试 4/4 通过（WSL + torch 2.14 CPU 实跑）；9 个 shell 脚本通过 `bash -n`。

---

## 1. `flashlm/compression/hypernetwork.py`

### 1.1 `simplifed_gate`：init 期一次性 prior 偏移（计划五，MaskLLM Eq.10 同构）

```python
class simplifed_gate(nn.Module):
    def __init__(self, t_structures, num_groups=1, reinmax=False,
                 prior_scores=None, prior_alpha=0.0):
```

- 构造时若传入 `prior_scores` 且 `prior_alpha != 0`，对每个 `p_list[i]` 做一次性
  `self.p_list[i].add_(prior_alpha * prior_scores[i])`（`torch.no_grad()` 下）。
- 偏移烘焙进可学习参数，**state_dict 格式不变，checkpoint 自动携带 prior**，
  eval 不需要任何额外操作。
- 计划中的"不会被后续训练冲垮、风险最低"路径。

### 1.2 `hypernetwork`：forward 期 prior 偏移（计划五指定插入点）

```python
class hypernetwork(nn.Module):
    def __init__(self, ..., prior_scores=None, prior_alpha=0.0):
```

- prior 存为**非持久化 buffer**（`persistent=False`）：
  - state_dict 与无 prior 时完全一致 → 旧 checkpoint 照常加载；
  - 随 `.to(device)` 一起上 GPU，forward 无 per-step host→device 拷贝。
- 新增 `_apply_prior(tp_out)`，在 `tp_out = [self.linear_list_tp[i](tp_out[i]) ...]`
  之后、train/eval 分支之前插入，**forward() 与 hard_output() 共用同一处**
  （train/eval 行为一致，正是计划要求的插入点）：

```python
tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]
tp_out = self._apply_prior(tp_out)          # <-- 新增
if not self.training:
    ...
```

- eval 复现偏移的方式：用训练写出的 sidecar `prior_scores.pt`
  重新构造（见 `hf_ppl.py` 的 `--prior_scores_path`）。

---

## 2. `flashlm/compression/semi_pruning_helper.py`

### 2.1 prior 计算（计划五）

新增三个函数（插在 `log_inv_function` 之后）：

- **`compute_group_prior_score(weight, ex_dict, act_norm=None)`**
  - 档位0：`prior = weight.abs()`（零成本）
  - 档位1：传入 `act_norm` 时 `prior = weight.abs() * act_norm`（Wanda 式，
    需一次校准前向）
  - 按档 pooling 到 group 粒度后去均值归一化：
    `score = (pooled - pooled.mean()) / (pooled.std() + 1e-6)`
  - **关键修正**：`virtual_operation.forward` 有两条展开分支，pooling 公式
    必须按分支对齐（最初实现照抄计划伪代码只覆盖了 4-D 分支，单元测试
    抓到错位后修正）：
      - **4-D 分支**（`groups_in_dim>1` 且 `groups_out_dim>1`）：
        gate `(j,i)` 对应权重块
        `rows [j*g_out:(j+1)*g_out] × cols [i*g_in:(i+1)*g_in]`，
        pooling = `prior.view(G_out, g_out, G_in, g_in).mean(dim=(1,3))`
      - **1-D 分支**（`groups_in_dim==1` 或 `groups_out_dim==1`，
        **生产配置 1×256 走这条分支**）：
        gate `k` 对应 tile
        `row k//(in_dim//R) × cols [R*(k%(in_dim//R)) : +R]`，
        其中 `R = groups_in_dim*groups_out_dim`；
        pooling = `prior.view(out_dim, in_dim//R, R).mean(-1)`
        （要求 `in_dim % R == 0`，Llama-2-7b 的 4096/11008/12288 均满足，
        不满足时抛出明确 `ValueError`）

- **`collect_act_norms(model, calib_input_ids)`**（`@torch.no_grad()`）
  - forward hook 收集每个 `SemiSparseLinear` 输入通道的
    `sqrt(Σx²/n)`（Wanda 同款激活 L2 norm）；
  - 临时置 `mask_flag=False` 走稠密前向，结束后恢复原 flags 与 train/eval 状态；
  - 返回 `{id(owner_module): (in_dim,) tensor}`。

- **`compute_model_prior_scores(model, mode, calib_input_ids)`**
  - `mode='magnitude'|'wanda'`，按 `collect_info_reg.structures` 的顺序
    （即 gate vector 顺序）返回所有结构的 score 列表。

### 2.2 `collect_info_reg`：per-layer uniform 正则（计划四 Uniform vs Adaptive）

```python
collect_info_reg(model, p, lam, per_layer=False)
```

- `per_layer=False`（默认）：原有全局预算 log-ratio 正则，行为不变（Adaptive 组）。
- `per_layer=True`：每个结构各自 pin 到 `p`，`len(structures)` 项平均
  （保持与全局版相同的 loss 量级）；**组内 mask 放置仍可学，
  只锁死跨层分配**（Uniform 组）。正面回应 ProxSparse Appendix E。
- 新增 `layer_keep_rates(vectors)`：hard（>0.5）keep-rate per 结构，
  供跨层分配对比图使用。

---

## 3. `recipe_train_groupsparsity.py`

### 3.1 新增 CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--prior_mode` | `none` | `none` / `magnitude`（档位0）/ `wanda`（档位1） |
| `--prior_alpha` | `0.0` | 偏移强度；0 = 即使 mode 非 none 也关闭 |
| `--prior_n_samples` | `8` | wanda 档位1 的校准 batch 数 |
| `--uniform_alloc` | `false` | true → `collect_info_reg(per_layer=True)`，每层强制 p |
| `--flip_log_interval` | `50` | `[FLIP]` 翻转率打印间隔（steps） |
| `--alloc_log_interval` | `500` | per-layer keep-rate CSV 落盘间隔（steps） |

### 3.2 main() 中的 prior 流程

- 在 `collect_info_reg` / hypernetwork 构造之间计算 prior：
  - sidecar 路径 `out_dir/prior_scores.pt`，内容
    `{"scores": [...], "meta": {mode, alpha, n_samples, structures}}`；
  - **已存在则复用**（resume 拿到与首次完全相同的 tensor）；
  - 只在 rank0 落盘；`share_qk=True` 时明确跳过并提示（共享 Q/K gate 不支持）；
  - wanda 模式从训练 dataloader 取前 `prior_n_samples` 个 batch 做校准前向
    （之后训练继续，对 c4 流数据影响可忽略）。
- `simplifed_gate` / `hypernetwork` 构造时传入 `prior_scores, prior_alpha`；
  顺带补上 `simplifed_gate` 构造后的 `hn.T = T`（此前只有 hypernetwork 会设置）。
- `train_hn(...)` 调用与签名新增
  `batch_size / flip_log_interval / alloc_log_interval`。

### 3.3 训练循环新增诊断

- **`tokens: N`**（每条 loss 日志行，阶段三 x 轴）：
  `tokens_done = iter_num * batch_size * hn_block_size * world_size`
- **`[FLIP]`**（阶段五 T 诊断）：每 `flip_log_interval` 步比较相邻两次 hard mask
  （>0.5 二值化）的翻转率
  `flip_rate = mean(cur_bits != prev_bits)`。
  T=0.4 与 T=0.8 两个 run 的日志对比即可验证"低温放大 GRU 抖动"。
- **`[ALLOC]`**（阶段四）：每 `alloc_log_interval` 步追加写
  `out_dir/layer_allocation.csv`（header: `iter,l0,l1,...,mean`），
  直接画 Uniform vs Adaptive 的跨层分配图。

---

## 4. `hf_ppl.py`

### 4.1 N:M magnitude baseline（计划二）

- 新增 `apply_nm_prune(model, n=2, m=4)`：对 q/k/v/o/gate/up/down 投影，
  沿输入维每连续 m 个权重保留幅度最大的 n 个。
- main() 新增分支（在 `semi_evaluate` 之前，不需要 hn ckpt）：
  加载 bf16 模型 → 剪枝 → PPL 评估 → 可选 `save_hf_dir` 导出 HF 模型。
- 新 CLI：`--nm_prune` / `--nm_n`(2) / `--nm_m`(4)。

### 4.2 eval 期 prior 重挂（计划五）

- 新 CLI：`--prior_scores_path`。
- hypernetwork：加载 ckpt 前读 sidecar，按 meta 里的 alpha 重建偏移
  （构造 hypernetwork 时传入），并在结构与 `param_reg.structures` 不匹配时 assert。
- simple_gate：**拒绝重挂**（prior 已烘焙在 `p_list` 里，重复施加会翻倍），
  打印提示忽略该参数。
- simple_gate 分支补上 `hn.T = T`（与训练一致）。

---

## 5. `test_prior_pooling.py`（新增，纯 CPU 可跑）

计划五点名的单元测试：人工标记一个 block 为极大值 → 跑完整
pooling → 展开 → 还原链路，确认 mask 落在正确位置。

| 测试 | 覆盖 |
|---|---|
| `test_score_shape_and_normalization` | score 长度 = gate dim，零均值单位方差 |
| `test_marked_block_lands_on_correct_gate` | 4-D 分支：标记块 → argmax gate → 展开回原块 |
| `test_wanda_act_norm_shifts_argmax` | 档位1：act_norm 确实按输入维加权、能改变 argmax |
| `test_groups_in_dim_one_branch` | 1-D 分支（生产 1×256 配置走的路径）：tile 语义对齐 |

运行：`python test_prior_pooling.py`（或 pytest）。**这个测试抓到并促使修复了
1-D 分支的 pooling 错位——提交 prior 相关实验前请先跑它。**

---

## 6. 新增 sbatch 脚本（×9，均可直接 `sbatch`）

全部沿用 `srun_group.sh` 的约定：同 partition/conda env/HF cache、
`send_notification`、`torchrun`、绝对路径配置区、`--out_dir` 固定 run 目录
（eval sweep 可直接指向）。

| 脚本 | 阶段 | 内容 | 预算 |
|---|---|---|---|
| `srun_train_simple_gate.sh` | 一 | simplifed_gate 40k（hypernetwork 必要性消融，其余与现有 40k 基线一致） | 12h |
| `srun_train_uniform_alloc.sh` | 四 | `--uniform_alloc=true` 40k（Uniform vs Adaptive；对照组=现有 hypernetwork run） | 12h |
| `srun_train_c4_dense.sh` | 三 | C4 流式 40k，`save_interval=2000` 跑 token-vs-PPL 曲线（需联网拉 shard，`HF_HUB_OFFLINE=0`） | 12h |
| `srun_train_prior_simple_mag.sh` | 五 | 档位0 magnitude prior，simple_gate，20k | 8h |
| `srun_train_prior_simple_wanda.sh` | 五 | 档位1 wanda prior，simple_gate，20k | 8h |
| `srun_train_prior_hn_wanda.sh` | 五 | wanda prior 迁移到 hypernetwork，20k | 8h |
| `srun_train_T08_diag.sh` | 五(诊断) | `--T=0.8` 5k 短跑，看 `[FLIP]` 对比 T=0.4 | 4h |
| `srun_eval_sweep.sh` | 一/三/四/五 | 通用 ckpt→PPL sweep（见下） | 8h |
| `srun_eval_nm24.sh` | 二 | 2:4 magnitude baseline 评估 + HF 导出 | 2h |

### `srun_eval_sweep.sh` 用法

```bash
# 默认: 自动选 outputs/groupsparsity 下最新 run, 评其全部 hn-ckpt-iter-*.pt (wikitext,ptb)
sbatch srun_eval_sweep.sh

# 指定 run / 步数 / 数据集 / simple_gate:
sbatch --export=CKPT_DIR=/path/to/run,STEPS_LIST="2000 4000 6000",DATASET=wikitext,SIMPLE_GATE=true srun_eval_sweep.sh
```

- hypernetwork 的 run 自动探测并挂回 `prior_scores.pt`；simple_gate 的 run 不需要。
- 每 ckpt 有 `EVAL_TIMEOUT=2h` 硬超时；结果汇总表按数据集列拆分
  （`ppl_results_<run>_<jobid>.log`）。

### 建议提交顺序

1. `python test_prior_pooling.py`（已验证通过）
2. `srun_train_T08_diag.sh` → `srun_train_prior_simple_mag.sh` / `_wanda.sh`（短跑先验证信号）
3. `srun_eval_nm24.sh`（随时可插队）
4. `srun_train_simple_gate.sh` → `srun_train_uniform_alloc.sh`（两个 12h 大跑）
5. `srun_train_c4_dense.sh`（需网，主证据线）
6. 各 run 完成后 `srun_eval_sweep.sh`（可用依赖：`sbatch --dependency=afterok:<jobid>`）

---

## 7. 已知边界 / 注意事项

- prior 计算在多卡下各 rank 独立算 act norm（wanda 档位1）；当前所有交付脚本
  均为单卡（`NPROC_PER_NODE=1`），无影响；若扩展多卡需改为 rank0 计算后广播。
- `[FLIP]` 的 `_prev_hard_bits` 常驻 GPU，大小 = 全部 gate bit 数（Llama-2-7b 约 176k×N 结构，可忽略）。
- `simplifed_gate` 的 prior 是 init 期烘焙，因此其 run 的 eval **不带也不需要** sidecar；
  `hypernetwork` 的 run 若无 sidecar（`prior_alpha=0`），eval 行为与旧版完全一致。
- 所有新参数默认值均保持旧行为（`prior_mode=none`、`uniform_alloc=false`、
  `nm_prune=false`），不传新参时训练/评估路径与改动前逐字节等价（除新增日志行）。
