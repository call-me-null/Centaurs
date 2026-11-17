# 太阳系小天体长期演化模拟（REBOUND 进程池版）

用 [REBOUND](https://rebound.readthedocs.io/) 模拟巨行星区域内测试粒子 1 亿年的轨道演化，追踪每个粒子的轨迹和去向。
用同一份代码分别计算 **完整太阳系** 和 **只有 4 颗巨行星** 两种情况，比较行星组成对结果的影响。

主程序：`centaurs.py`

---

## 模型设定

| 项目 | 设定 |
|---|---|
| 天体组合 `full` | 太阳 + 8 大行星 + 5 颗矮行星（Ceres、Pluto、Eris、Haumea、Makemake） |
| 天体组合 `giants` | 太阳 + 木星、土星、天王星、海王星（对照组） |
| 初始历元 | 2024-01-01 00:00（JPL Horizons） |
| 单位 | AU、yr、Msun（G = 4π²） |
| 积分器 | TRACE（混合积分器），步长 dt = 0.012 yr（约为水星周期的 1/20），两组相同 |
| 积分时长 / 快照间隔 | 1×10⁸ yr / 1×10⁵ yr（共 1000 个快照） |
| 测试粒子 | 无质量（`N_active` = 大天体数） |

### 初始粒子

4 个区域，每个区域 128 个粒子，共 512 个：

| 区域 | 近日距 q 的范围 | pid |
|---|---|---|
| JS | a_J – a_S | 0 – 127 |
| SU | a_S – a_U | 128 – 255 |
| UN | a_U – a_N | 256 – 383 |
| NK | a_N – 40 AU | 384 – 511 |

- q 在区域内均匀随机抽取；远日距 Q = 40 AU，所以 a = (q + Q)/2，e = 1 − q/a
- inc = 0.05 rad，Ω = ω = M = 0
- 初始根数相对太阳（日心）给定
- 随机数由 `seed + 任务编号` 决定，两种天体组合中**同一 pid 的初始 q 完全相同**，可以逐个粒子配对比较

### 粒子去向

| 代码 | 名称 | 判定 |
|---|---|---|
| 0 | survived | 存活到积分结束 |
| 1 | ejected | 距质心 r > 1000 AU，被移除 |

**不做碰撞检测**，所有天体都视为质点。

---

## 环境

- Python 3.9+
- `rebound >= 4.4`（TRACE 积分器需要；推荐 5.x，代码兼容 4.x 和 5.x）
- `numpy`、`astropy`

```bash
pip install rebound numpy astropy
```

> 同一次计算（包括中断后续算）必须始终使用同一个 REBOUND 版本和编译方式（例如自己编译的 FMA 版本）。
> 不同的编译方式会改变浮点舍入，续算结果就不再与不中断时逐位一致。

---

## 使用方法

### 1. 建立初始天体系统

```bash
# 完整组：查询 Horizons（只有这一步需要联网），检查打印出的各天体质量都不为 0
python centaurs.py build-base --bodies full

# 对照组不必单独建立：正式运行时若 results_giants 中没有 base 文件，
# 会自动从 results_full 的文件中挑出太阳 + 4 巨行星（不联网，保证初始状态一致）。
# 也可以手动建立：
python centaurs.py build-base --bodies giants --source-base results_full/base_solar_system.bin
```

### 2. 开跑前的检查（建议）

```bash
# 测速：日志中单个任务的用时 × 1000 ≈ 跑满 1e8 年的用时
python centaurs.py run --t-end 1e5 --dt-out 1e4 --n-per-region 40 --out-dir bench_trace

# 续算演练：中途 Ctrl-C，再执行同一条命令，日志中应出现 resumed+done
python centaurs.py run --t-end 2e4 --dt-out 100 --n-per-region 8 --out-dir resume_test

# 收敛测试（先把同一个 base_solar_system.bin 复制到各目录）
python centaurs.py run --integrator trace --n-per-region 40 --t-end 1e7 --out-dir conv_trace
python centaurs.py run --integrator trace --dt 0.006 --n-per-region 40 --t-end 1e7 --out-dir conv_trace_half
python centaurs.py run --integrator ias15 --n-per-region 40 --t-end 1e7 --out-dir conv_ias15
```

### 3. 正式运行

在 tmux 中运行，防止 SSH 断开时进程被结束：

```bash
tmux new -s sim
nice -n 19 python centaurs.py run --bodies full giants
# Ctrl-B 再按 D 离开会话；tmux attach -t sim 回来查看
```

两组的任务放进**同一个进程池**：按"长任务优先"混合提交，一组的长任务还在运行时，空出来的核会立即去算另一组的任务，不需要中途再执行命令。
两组仍写入各自的目录和日志；某一组的任务全部结束时，这一组立即自动合并（另一组继续运行）。

中途查看已完成的部分：

```bash
python centaurs.py merge --bodies full giants
```

### 4. 中断后续算

用**完全相同的参数**再执行一次同一条命令即可：已完成的任务跳过，未完成的任务从 checkpoint 继续。
除 `--workers` 外，参数、天体组合或 `base_solar_system.bin` 只要有变化，程序就会报错并拒绝运行，防止新旧数据混在一起。

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `run` / `merge` / `build-base` | `run` | 运行 / 合并 / 建立初始天体系统 |
| `--bodies` | `full` | `full`、`giants`，可同时指定多个（共用一个进程池） |
| `--out-prefix` | `results` | 输出目录前缀，目录为 `<prefix>_<bodies>` |
| `--out-dir` | — | 直接指定输出目录（只能配合单个天体组合） |
| `--source-base` | — | `build-base` 时从已有文件挑选天体 |
| `--workers` | 40 | 进程池大小 |
| `--particles-per-task` | 4 | 每个任务的粒子数 |
| `--n-per-region` | 128 | 每个区域的粒子数（试跑时可调小） |
| `--integrator` | `trace` | `trace` / `mercurius` / `ias15` |
| `--dt` | 0.012 | 步长 [yr]（trace / mercurius 使用） |
| `--t-end` | 1e8 | 积分时长 [yr] |
| `--dt-out` | 1e5 | 快照间隔 [yr] |
| `--seed` | 20240101 | 随机种子 |

其他参数在代码中的 `DEFAULTS` 里设置：`ckpt_every = 10`（每 10 个快照写一次 checkpoint，即每 1e6 yr）、`exit_max_distance = 1000` AU、`Q = 40` AU、`inc = 0.05` rad、`random_angles = False`。

---

## 并行与调度

- 粒子按每个任务 4 个切分，交给 `multiprocessing.Pool` 动态分配；一个任务结束，空出的核立刻接下一个任务
- 任务内的粒子全部被抛出后，任务提前结束（内侧区域的任务通常很短）
- **长任务优先**：按 NK → UN → SU → JS 的顺序提交，短任务放在最后填补空闲的核
- 多个天体组合（`--bodies full giants`）的任务放进同一个进程池混合调度：先按区域，同一区域内 full 在前
- 每个任务都在全新的进程中运行（`maxtasksperchild=1`）；任务出错只影响它自己，错误信息会写进日志

---

## 输出文件

```
results_full/                      （results_giants/ 结构相同）
├── base_solar_system.bin          初始天体状态（REBOUND 二进制格式）
├── run_log.txt                    运行日志（只记录本组）：开始、每个任务完成、错误、总用时、合并统计
├── all_particles.npz              全部任务的合并结果（分析主要用这个）
└── tasks/
    ├── task_00000.npz …           每个任务的结果
    ├── task_XXXXX.partial.npz     ┐ 运行中的 checkpoint，任务完成后自动删除
    └── task_XXXXX.ckpt.bin        ┘
```

### `all_particles.npz` 的内容

npz 是 numpy 的格式，一个文件里装着多个有名字的数组，用 `np.load` 读取。

**每个粒子的数据**（第 0 维是粒子，按 pid 排序，`数组[pid]` 就是这个粒子的数据）：

| 字段 | 形状 | 内容 |
|---|---|---|
| `pid`, `region`, `task_id` | (512,) | 粒子编号、所属区域、所属任务 |
| `q0`, `a0`, `e0`, `inc0`, `Omega0`, `omega0`, `M0` | (512,) | 初始日心轨道根数 |
| `elems` | (512, 1000, 7) | 每个快照的日心根数：a, e, inc, Ω, ω, M, r |
| `xv` | (512, 1000, 6) | 每个快照的日心状态矢量：x, y, z, vx, vy, vz |
| `fate` | (512,) | 0 = 存活，1 = 被抛出 |
| `fate_time` | (512,) | 被抛出的时刻（精确到积分步）；存活粒子为最后一个快照的时刻 |
| `fate_xv` | (512, 6) | 被抛出瞬间的日心状态矢量；存活粒子为最后一个快照的状态 |

粒子被移除后，`elems` 和 `xv` 中对应的位置填 NaN。

**每个任务的数据**（第 0 维是任务，顺序与 `task_ids` 对应）：

| 字段 | 形状 | 内容 |
|---|---|---|
| `task_ids` | (128,) | 任务编号 |
| `bodies_xv` | (128, N_天体, 1000, 6) | 大天体的质心系状态矢量（full 组 14 个，giants 组 5 个） |
| `dE` | (128, 1000) | 相对能量误差（只反映大天体的积分，见下文） |

**公共数据**：

| 字段 | 内容 |
|---|---|
| `t_start` | 初始时刻 [yr，从 J2000 起算] |
| `t_out` | (1000,) 快照的名义时刻 = t_start + (j+1)·dt_out |
| `fields`, `fields_xv` | `elems` 和 `xv` 最后一维的字段名 |
| `body_labels` | `bodies_xv` 中天体的名称和顺序 |
| `config` | 本次运行的全部参数（JSON 字符串） |

单个任务文件 `task_XXXXX.npz` 的字段基本相同，只包含该任务的 4 个粒子；另外有 `E0`（初始能量）和 `snap_done`（已完成的快照数）。

### 合并方式

时间轴和数据种类在所有任务中完全相同，因此合并时沿粒子维度用 `np.concatenate` 拼接，再按 pid 排序；没有粒子维度的 `bodies_xv` 和 `dE` 则用 `np.stack` 增加一个任务维度。

---

## 读取示例

```python
import numpy as np, json

d = np.load("results_full/all_particles.npz")
cfg = json.loads(str(d["config"]))

d["elems"][37, :, 0]          # 粒子 37 的 a 随时间的变化
d["elems"][:, 499, :]         # 第 500 个快照（约 5000 万年）所有粒子的根数
d["xv"][37, :, :3]            # 粒子 37 的日心位置轨迹
d["t_out"] - d["t_start"]     # 从模拟开始算起的时间

ejected = d["pid"][d["fate"] == 1]
lifetime = d["fate_time"] - d["t_start"]

# 粒子所在任务的行星轨迹
i = np.where(d["task_ids"] == d["task_id"][37])[0][0]
planets = d["bodies_xv"][i]   # (N_天体, 1000, 6)

# 两种天体组合的逐粒子比较（同一 pid 的初始 q 相同）
f = np.load("results_full/all_particles.npz")
g = np.load("results_giants/all_particles.npz")
assert np.array_equal(f["q0"], g["q0"])
```

导出成 CSV（例如第 500 个快照）：

```python
import pandas as pd
df = pd.DataFrame(d["elems"][:, 499, :], columns=d["fields"])
df.insert(0, "pid", d["pid"])
df.to_csv("snapshot_0500.csv", index=False)
```

---

## 注意事项

- **不做碰撞检测**：粒子可以从行星中心附近穿过，或者以极小的近日距掠过太阳，从而获得非物理的强烈引力弹弓，有些粒子会因此被"抛出"。分析时可以用快照中的近日距 q = a(1−e)，以及与行星的距离来甄别。
- **"被抛出"的定义**：r > 1000 AU 不一定是双曲轨道逃逸，也可能是被散射到很远但仍受束缚的轨道。需要区分时，可以用 `fate_xv` 计算日心能量。
- **轨道根数是日心的**：外太阳系天体的日心根数会随太阳绕质心的摆动（约 12 年周期）小幅波动；也可以用 `xv` 和 `bodies_xv` 计算质心根数或 Jacobi 根数。
- **`dE` 只反映大天体（行星）的积分**：测试粒子质量为 0，不计入能量。粒子轨道的精度需要靠收敛测试（步长减半、与 IAS15 对比）来验证。
- **不同任务中的行星轨迹会逐渐分叉**：测试粒子的近距离交会会改变积分器在该步的处理方式，产生舍入级别的差异，经过混沌放大后，各任务中的行星位置最终会不同。因此每个任务的行星轨迹都单独保存。
- **快照时刻**：TRACE 不强制停在整点，实际时刻比 `t_out` 晚不到一个步长（< 0.012 yr）。
- **不依赖粒子名称**：REBOUND 5.2.2 在同一进程中"保存 → 释放 → 读取"文件后，读回的粒子名称可能是错的（文件本身正确）。因此计算中测试粒子用 `sim_order`（模拟中粒子顺序 → pid，随 checkpoint 保存）识别，天体名称的检查一律在新的干净进程中进行。
- **可重复性**：同一台机器、同一个 REBOUND 版本下，结果逐位可重复（包括中断后续算）；不同的 CPU 架构或编译方式下，结果只在统计上一致。