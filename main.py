# -*- coding: utf-8 -*-
"""
main.py —— 模拟配置（只放参数；计算代码在 centaurs.py，合并在 merge.py）

改这个文件来决定"算什么"：太阳系构成、粒子初始条件、每种组合的任务数量和启动顺序、进程数。
三个文件放在同一目录下。

用法
-----
    python main.py plan                       # 查看任务分配表（不计算）
    python main.py build-base                 # 查询 Horizons，检查打印出的质量
    tmux new -s sim
    nice -n 19 python main.py run             # 正式运行，输出到 results_<今天日期>/
    # 续算、追加批次：必须写上原来的日期
    nice -n 19 python main.py run --date 20261121
    python main.py plan --date 20261121       # 查看完成情况
    # 全部算完后合并（可随时运行，只合并已完成的任务）
    python merge.py results_20261121
    python merge.py results_20261121 --series # 同时合并完整时间序列（文件很大）

其他选项（只筛选本次运行的任务，不改变编号和结果）
    --only-bodies full giants   --only-regions NK UN   --limit-tasks 2   --workers 44
    --root 目录名                直接指定总目录（试跑用）

试跑 / 测速 / 收敛测试：把本文件复制一份（例如 main_test.py），改短 SIM["t_end"]、改小 BATCHES，
再用 python main_test.py run --root bench 运行。改变结果的参数只在配置文件中设置，不在命令行覆盖。
可以把正式目录中的 base_horizons_all.bin 复制到试跑目录，免得再查 Horizons。
"""
import math

import centaurs

# =============================================================================
# 起始历元
# =============================================================================
START_DATE = "2024-01-01 00:00"

a_J = 5.2023651711779895
a_S = 9.545907027243075
a_U = 19.19313007189107
a_N = 30.077369437624583

MSUN_KG = 1.98847e30

# =============================================================================
# 天体（完整列表，Horizons 只查询一次，各种构成都从中挑选）
# (Horizons 查询字符串, 标签, 质量[Msun] 或 None=用 Horizons/REBOUND 自带值)
# 小天体在 Horizons 中用 "编号;" 查询；矮行星的质量 Horizons 往往不给，这里手动指定。
# =============================================================================
BODIES = [
    ("Sun",     "Sun",      None),
    ("Mercury", "Mercury",  None),
    ("Venus",   "Venus",    None),
    ("Earth",   "Earth",    None),
    ("Mars",    "Mars",     None),
    ("Jupiter", "Jupiter",  None),
    ("Saturn",  "Saturn",   None),
    ("Uranus",  "Uranus",   None),
    ("Neptune", "Neptune",  None),
    ("1;",      "Ceres",    9.3839e20 / MSUN_KG),
    ("9",       "Pluto",    1.4705e22 / MSUN_KG),   # 冥王星系统质心（含卡戎）
    ("136199;", "Eris",     1.6466e22 / MSUN_KG),
    ("136108;", "Haumea",   4.006e21 / MSUN_KG),
    ("136472;", "Makemake", 3.1e21 / MSUN_KG),
]

# =============================================================================
# 太阳系构成
# 名称用作目录名（results_<名称>），只能用字母、数字、下划线、连字符；
# 第一个必须是 Sun；天体顺序与 BODIES 一致。
# =============================================================================
ALL_BODIES = [b[1] for b in BODIES]
GIANTS = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"]
DWARFS = ["Ceres", "Pluto", "Eris", "Haumea", "Makemake"]

BODY_SETS = {
    "full":          ALL_BODIES,          # 太阳 + 8 行星 + 5 矮行星
    "giants":        GIANTS,              # 太阳 + 4 颗巨行星
    "giants_dwarfs": GIANTS + DWARFS,     # 太阳 + 4 颗巨行星 + 5 矮行星
}

# =============================================================================
# 粒子初始条件（区域）
# id      : 固定编号，决定 pid 段（pid = id × 1,000,000 + 区域内序号）和随机数流。
#           一经使用不要修改；新增区域用新的 id。
# q_min, q_max : 近日距 q 在此范围内均匀抽取（AU）
# Q       : 远日距（AU）；a = (q + Q) / 2, e = 1 - q / a
# inc     : 倾角（rad）
# Omega, omega, M : 范围 (low, high)，在其中均匀抽取（rad）；low == high 即固定值
#                   例：(0, TWO_PI) 在 0–2π 内均匀分布；(0.0, 0.0) 全取 0；(-0.1, 0.1) 小扰动
# 每个粒子依次抽取 q, Omega, omega, M；若初始位置在任一行星或矮行星的 hill_factor × 希尔半径以内
# （见 SIM["hill_factor"]），就重新抽取。检查对 BODIES 中太阳以外的全部天体进行，不论构成。
# 每个粒子的初始条件只由 (seed, 区域 id, 区域内序号) 决定，与任务怎么切分、属于哪种构成无关
# → 同一 pid 在各构成中初始条件完全相同。
# =============================================================================
TWO_PI = 2.0 * math.pi
ANG = dict(Omega=(0.0, TWO_PI), omega=(0.0, TWO_PI), M=(0.0, 0.0))

REGIONS = {
    "JS": dict(id=0, q_min=a_J, q_max=a_S, Q=40.0, inc=0.05, **ANG),
    "SU": dict(id=1, q_min=a_S, q_max=a_U, Q=40.0, inc=0.05, **ANG),
    "UN": dict(id=2, q_min=a_U, q_max=a_N, Q=40.0, inc=0.05, **ANG),
    "NK": dict(id=3, q_min=a_N, q_max=40.0, Q=40.0, inc=0.05, **ANG),
}

# =============================================================================
# 任务批次（数量 + 启动顺序）
# 每行: (区域, 任务数, 每任务粒子数, [太阳系构成, ...])
#   该行粒子数 = 任务数 × 每任务粒子数，在列出的每种构成中各算一次（同一 pid 可跨构成配对）。
# 启动顺序：按行从上到下提交；行内按任务交替提交各构成
#   （NK_000000 full → giants_dwarfs → giants → NK_000008 full → ...）。
#   建议预计寿命长的区域、天体多的构成放在前面（长任务优先，缩短最后的"拖尾"）。
# 编号：同一区域的粒子按该区域各行的先后连续编号。
#   * 不同区域的行之间可以随意调换（只改变启动顺序，不影响编号）；
#   * 同一区域的多行之间不要调换；已运行过的行不要修改或删除；
#   * 追加粒子：加新行；给已有的行增加一种构成也可以。
#   改错时程序会检测到与已有结果冲突并拒绝运行。
# =============================================================================
#ALL_SETS = ["full", "giants_dwarfs", "giants"]
ALL_SETS = ["giants_dwarfs", "giants"]
BATCHES = [
    ("NK", 128, 8, ALL_SETS),
    ("UN", 128, 8, ALL_SETS),
    ("SU", 128, 8, ALL_SETS),
    ("JS", 128, 8, ALL_SETS),
]

# =============================================================================
# 积分参数（影响结果；一次运行中不能改变，续算时会比对）
# =============================================================================
SIM = dict(
    seed=20240101,
    # 积分器："trace"（推荐，REBOUND>=4.4，能处理近距离交会和近日点小的轨道）
    #        "mercurius"（较老的混合积分器）  "ias15"（最精确但极慢，只用于收敛测试）
    integrator="trace",
    dt=0.012,                   # 年；约为水星周期的 1/20。trace/mercurius 才用
    t_end=1e8,                  # 积分总时长（年，从起始历元算起）
    dt_out=1e4,                 # 快照间隔（年）
    exit_max_distance=10000.0,  # AU，距质心超过即视为被抛出
    hill_factor=1.0,            # 粒子初始位置须在每个天体的 hill_factor × 希尔半径以外；0 = 不检查
)

# =============================================================================
# 运行参数（不影响结果）
# =============================================================================
RUN = dict(
    workers=44,                 # 进程池同时运行的进程数
    ckpt_every=20,              # 每多少个快照写一次 checkpoint（20 × 1e4 年 = 2e5 年）
    out_prefix="results",       # 总目录 <prefix>_<日期>，构成目录 <prefix>_<构成>
)


if __name__ == "__main__":
    centaurs.cli(dict(START_DATE=START_DATE, BODIES=BODIES, BODY_SETS=BODY_SETS,
                      REGIONS=REGIONS, BATCHES=BATCHES, SIM=SIM, RUN=RUN),
                 config_file=__file__)