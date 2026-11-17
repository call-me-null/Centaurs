# -*- coding: utf-8 -*-
"""
太阳系小天体长期演化模拟 —— 进程池版本

概要
-----
1. 进程池：所有粒子按每任务 PARTICLES_PER_TASK(=4) 个切成小任务，交给
   multiprocessing.Pool（默认 40 个进程）动态分配。某个任务跑完，空出的核立刻接下一个。
   任务内所有粒子都被抛出后提前结束。调度采用"长任务优先"：NK、UN 区域先提交。
2. 天体组合（--bodies，可同时指定多个）：
     full   : 太阳 + 8 大行星 + 5 颗 IAU 矮行星（Ceres、Pluto、Eris、Haumea、Makemake）
     giants : 太阳 + 4 颗巨行星（对照组）
   Horizons 只在主进程查询一次（完整列表），存成 base_solar_system.bin，所有子进程只读这个文件。
   giants 组的 base 文件自动从 full 组的 base 文件中挑选（或用 --source-base 指定），两组巨行星的日心初始状态
   相同（仅差重新移到质心系时的舍入误差，约 1e-15）。两组使用相同的随机种子，因此同一 pid 的粒子初始 q 完全相同，可逐个配对比较。
3. 不做碰撞检测：所有天体都视为质点，粒子只有两种去向：
         0 = survived  存活到结束
         1 = ejected   距质心 r > EXIT_MAX_DISTANCE（1000 AU）即视为被抛出并移除
   注意：质点之间没有碰撞，粒子可以从行星"中心附近"穿过并获得非物理的强烈引力弹弓，
   也可能以极小近日距掠过太阳。这类情况需要在分析时由快照的 a, e（近日距 q）或
   与行星的距离自行甄别。
4. 每个粒子有全局唯一编号 pid，每个快照记录
     - 日心轨道根数 a, e, inc, Omega, omega, M, r          -> elems (粒子, 快照, 7)
     - 日心状态矢量 x, y, z, vx, vy, vz（AU, AU/yr）       -> xv    (粒子, 快照, 6)
   每个任务另存全部大天体的质心系状态矢量 bodies_xv 和相对能量误差 dE
   （dE 只反映行星积分；测试粒子质量为 0 不计入能量，粒子精度需靠收敛测试验证）。
   被抛出的时刻精确到积分步，事件瞬间的日心状态存为 fate_xv。
5. 输出文件（只有以下几种）：
     results_<bodies>/base_solar_system.bin        Horizons 下载的初始天体状态
     results_<bodies>/tasks/task_XXXXX.npz         每个任务的结果
     results_<bodies>/tasks/task_XXXXX.partial.npz ┐ 运行中的 checkpoint，任务完成后自动删除
     results_<bodies>/tasks/task_XXXXX.ckpt.bin    ┘
     results_<bodies>/all_particles.npz            全部任务完成后自动合并的结果
     results_<bodies>/run_log.txt                  运行日志（开始、每个任务完成、出错信息、总用时）
   运行参数以 JSON 字符串形式存在每个 npz 的 "config" 字段中，续算前会逐一比对，
   参数（除进程数外）、天体组合或 base_solar_system.bin 有任何变化都会拒绝运行，防止数据混杂。
6. 断点续算：每 CKPT_EVERY 个快照（默认 10 个 = 1e6 年）写一次 checkpoint；中断后用相同参数
   重新运行，已完成任务跳过，未完成任务从 checkpoint 继续，结果与不中断时逐位一致。

用法
-----
    # 1) 完整组：查询 Horizons，检查打印出的质量
    python centaurs.py build-base --bodies full
    # 2) 正式运行：两组放进同一个进程池（在 tmux 中运行），每组的任务全部结束后各自自动合并。
    #    giants 组的 base 文件若不存在，会自动从 results_full 的 base 文件挑选生成（不联网）
    tmux new -s sim
    nice -n 19 python centaurs.py run --bodies full giants
    # 手动合并当前已完成的任务（中途查看用）
    python centaurs.py merge --bodies full giants

多个天体组合共用一个进程池：任务按"长任务优先"混合提交（NK → UN → SU → JS，同一区域内 full 在前），
一组的长任务还在运行时，空出来的核会立即去算另一组的任务。各组仍写入各自的目录、日志，分别合并。

开跑前的检查
    # 测速：单任务用时 × 1000 ≈ 跑满 1e8 年的用时
    python centaurs.py run --t-end 1e5 --dt-out 1e4 --n-per-region 40 --out-dir bench_trace
    # 续算演练：中途 Ctrl-C，再执行同一命令，应出现 resumed+done
    python centaurs.py run --t-end 2e4 --dt-out 100 --n-per-region 8 --out-dir resume_test
    # 收敛测试（可先缩短 --t-end）：比较三组的去向比例与寿命分布
    #   先把同一个 base_solar_system.bin 复制到各目录，保证初始条件完全一致
    python centaurs.py run --integrator trace --n-per-region 40 --t-end 1e7 --out-dir conv_trace
    python centaurs.py run --integrator trace --dt 0.006 --n-per-region 40 --t-end 1e7 --out-dir conv_trace_half
    python centaurs.py run --integrator ias15 --n-per-region 40 --t-end 1e7 --out-dir conv_ias15

依赖：rebound >= 4.4（推荐 5.x），numpy，astropy
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import re
import json
import hashlib
import time
import argparse
import traceback
from datetime import datetime
from multiprocessing import Pool, get_context

import numpy as np
import rebound
from astropy.time import Time


# =============================================================================
# 配置
# =============================================================================
START_DATE = "2024-01-01 00:00"

a_J = 5.2023651711779895
a_S = 9.545907027243075
a_U = 19.19313007189107
a_N = 30.077369437624583

# (区域名, q 下限, q 上限, 该区域总粒子数)
# 原代码每个区域 4 组 × 2 进程 × 16 粒子 = 128 个
# pid 按此顺序连续编号：JS 0-127, SU 128-255, UN 256-383, NK 384-511
REGIONS = [
    ("JS", a_J, a_S, 128),
    ("SU", a_S, a_U, 128),
    ("UN", a_U, a_N, 128),
    ("NK", a_N, 40.0, 128),
]

DEFAULTS = dict(
    bodies="full",       # 天体组合，见 BODY_SETS
    out_dir=None,        # None -> results_<bodies>
    workers=40,
    particles_per_task=4,
    seed=20240101,
    # 积分器："trace"（推荐，REBOUND>=4.4，能处理近距离交会和近日点小的轨道）
    #        "mercurius"（较老的混合积分器）  "ias15"（最精确但极慢，只用于收敛测试）
    integrator="trace",
    dt=0.012,            # 年；约为水星周期的 1/20。trace/mercurius 才用
    t_end=1e8,           # 积分总时长（年，从起始历元算起）
    dt_out=1e5,          # 快照间隔（年）
    ckpt_every=10,       # 每多少个快照写一次 checkpoint（10 × 1e5 年 = 1e6 年，约 15 分钟）
    exit_max_distance=1000.0,   # AU，超过即视为被抛出
    Q=40.0,              # 远日距
    inc=0.05,            # rad
    random_angles=False, # False（默认）: Omega, omega, M 全取 0，与原代码一致；True: 均匀随机
)

MSUN_KG = 1.98847e30

# (Horizons 查询字符串, 标签, 质量[Msun] 或 None=用 Horizons/REBOUND 自带值)
# 小天体在 Horizons 中用 "编号;" 查询；矮行星的质量 Horizons 往往不给，这里手动指定。
# 不做碰撞检测，所有天体都是质点，不设半径。
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
BODY_LABELS = [b[1] for b in BODIES]

# 天体组合（--bodies）。初始状态总是先按完整列表查询 Horizons，再从中挑选，
# 因此不同组合中同一天体的日心初始状态相同（仅差舍入误差）。
BODY_SETS = {
    "full":   BODY_LABELS,                                          # 太阳 + 8 行星 + 5 矮行星
    "giants": ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"],     # 太阳 + 4 颗巨行星（对照组）
}


def body_labels(cfg):
    return BODY_SETS[cfg["bodies"]]

FIELDS = ("a", "e", "inc", "Omega", "omega", "M", "r_helio")
# 状态矢量：单位 AU 和 AU/yr（因为 sim.units = AU, yr, Msun）
FIELDS_XV = ("x", "y", "z", "vx", "vy", "vz")
FATE_SURVIVED, FATE_EJECTED = 0, 1
FATE_NAMES = {0: "survived", 1: "ejected"}


# =============================================================================
# REBOUND 4 / 5 兼容小工具（5.x 用 name 字符串，4.x 用 hash）
# =============================================================================
_HAS_NAME = hasattr(rebound.Particle, "name")


def set_label(p, label):
    if _HAS_NAME:
        p.name = label
    else:
        p.hash = label


def get_label(p, hash_table):
    if _HAS_NAME:
        return p.name
    return hash_table.get(p.hash.value, "?")


def _read_labels(path):
    sim = rebound.Simulation(path)
    table = make_hash_table(BODY_LABELS)
    return [get_label(sim.particles[i], table) for i in range(sim.N)]


def read_labels_clean(path):
    """在一个全新的进程（spawn）中读取 base 文件里的天体名称。
    REBOUND 5.2.2 在同一进程中"保存→释放→读取"后，读回的名称可能是错的，
    所以名称检查一律放在干净的进程里做。"""
    with get_context("spawn").Pool(1) as pool:
        return pool.apply(_read_labels, (path,))


def make_hash_table(labels):
    if _HAS_NAME:
        return {}
    return {rebound.hash(s).value: s for s in labels}


def integrator_name(sim):
    integ = sim.integrator
    return getattr(integ, "name", integ)


def tp_label(pid):
    return f"tp{int(pid)}"


# =============================================================================
# 路径
# =============================================================================
def paths(cfg):
    out = cfg["out_dir"]
    return dict(
        base=os.path.join(out, "base_solar_system.bin"),
        tasks=os.path.join(out, "tasks"),
        merged=os.path.join(out, "all_particles.npz"),
        log=os.path.join(out, "run_log.txt"),
    )


def task_files(cfg, tid):
    d = paths(cfg)["tasks"]
    stem = os.path.join(d, f"task_{tid:05d}")
    return dict(final=stem + ".npz", ckpt_npz=stem + ".partial.npz", ckpt_sim=stem + ".ckpt.bin")


# =============================================================================
# 1. 主进程：建立行星系统（只做一次）
# =============================================================================
def build_base_simulation(cfg, source_base=None):
    """建立初始天体系统并保存。
    source_base=None : 查询 Horizons（完整天体列表），再挑出 --bodies 指定的天体
    source_base=路径 : 从已有的 base 文件（通常是完整组的）中挑出指定天体，不联网
    """
    p = paths(cfg)
    wanted = body_labels(cfg)
    if source_base:
        print(f"Deriving base simulation from {source_base} ...")
        have = read_labels_clean(source_base)
        missing = [lb for lb in wanted if lb not in have]
        if missing:
            sys.exit(f"Error: {source_base} does not contain {missing}")
        sim = rebound.Simulation(source_base)
    else:
        sim = rebound.Simulation()
        sim.units = ("AU", "yr", "Msun")
        print("Querying JPL Horizons ...")
        for query, label, mass in BODIES:
            # 注意：必须传 date，否则 Horizons 返回的是"运行当下"的位置
            sim.add(query, date=START_DATE)
            part = sim.particles[-1]
            if mass is not None:
                part.m = mass
            set_label(part, label)
        have = [b[1] for b in BODIES]
        jd = Time(START_DATE.replace(" ", "T"), format="isot", scale="utc").jd
        # 单位为 (AU, yr, Msun) 时时间单位就是"年"，不需要再乘 2π
        sim.t = (jd - 2451545.0) / 365.25

    # 只保留指定天体（按索引倒序删除，不读取粒子名称）
    for i in range(sim.N - 1, -1, -1):
        if have[i] not in wanted:
            sim.remove(i)
            del have[i]
    if have != wanted:
        sys.exit(f"Error: unexpected body order {have}, expected {wanted}")
    # 天体组成变了，系统质心也变了，重新移到质心系（不改变各天体的日心状态）
    sim.move_to_com()

    if os.path.exists(p["base"]):
        os.remove(p["base"])
    sim.save_to_file(p["base"])
    if read_labels_clean(p["base"]) != wanted:            # 在干净进程中复核
        sys.exit(f"Error: verification of {p['base']} failed")

    print(f"Base simulation saved to {p['base']}  (bodies = {cfg['bodies']}, N = {sim.N}, "
          f"t0 = {sim.t:.4f} yr since J2000)")
    for i, label in enumerate(wanted):
        part = sim.particles[i]
        a = part.orbit(primary=sim.particles[0]).a if i else 0.0
        print(f"  {label:9s} m = {part.m:.4e} Msun   a_helio = {a:.4f} AU")
        if part.m == 0:
            print(f"  WARNING: {label} has zero mass!")


# =============================================================================
# 2. 任务划分
# =============================================================================
# 提交顺序：预计寿命最长的区域先算（长任务优先调度）
SCHEDULE_ORDER = ["NK", "UN", "SU", "JS"]


def make_tasks(cfg):
    """按 REGIONS 顺序切分任务并编号（task_id、pid 连续），再按 SCHEDULE_ORDER 排列提交顺序。"""
    ppt = cfg["particles_per_task"]
    tasks, tid, pid = [], 0, 0
    for region, q1, q2, n_total in REGIONS:
        n_total = cfg.get("n_per_region") or n_total
        for start in range(0, n_total, ppt):
            n = min(ppt, n_total - start)
            tasks.append(dict(task_id=tid, region=region, q_min=q1, q_max=q2,
                              n=n, first_pid=pid, cfg=cfg))
            tid += 1
            pid += n
    priority = {r: k for k, r in enumerate(SCHEDULE_ORDER)}
    tasks.sort(key=lambda t: (priority.get(t["region"], len(priority)), t["task_id"]))
    return tasks


# =============================================================================
# 3. 子进程：单个任务
# =============================================================================
def configure(sim, cfg):
    if integrator_name(sim) != cfg["integrator"]:
        sim.integrator = cfg["integrator"]
    if cfg["integrator"] in ("trace", "mercurius", "whfast"):
        sim.dt = cfg["dt"]
    sim.collision = "none"                    # 不做碰撞检测：所有天体视为质点
    sim.exit_max_distance = cfg["exit_max_distance"]


def helio_xv(p, sun):
    return (p.x - sun.x, p.y - sun.y, p.z - sun.z, p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz)


# 测试粒子的识别方式：sim_order[k] = 模拟中第 (N_active + k) 个粒子的 pid。
# 不依赖 REBOUND 的粒子名称：REBOUND 5.2.2 在同一进程中"保存→释放→读取"后，
# 读回的名称可能是错的（文件本身正确），因此计算中完全不从文件读回名称。
def remove_escapers(sim, events, rmax, sim_order):
    removed = 0
    r2 = rmax * rmax
    na = sim.N_active
    for i in range(sim.N - 1, na - 1, -1):     # 倒序删除，不影响前面的索引
        p = sim.particles[i]
        if p.x * p.x + p.y * p.y + p.z * p.z > r2:
            events.append((int(sim_order[i - na]), FATE_EJECTED, sim.t,
                           helio_xv(p, sim.particles[0])))
            sim.remove(i)                       # 保持顺序删除，与 sim_order 同步
            del sim_order[i - na]
            removed += 1
    return removed


def apply_events(events, state, col_of, alive):
    for pid, fate, t, xv in events:
        if pid in alive:                        # 同一粒子只记第一次
            k = col_of[pid]
            state["fate"][k] = fate
            state["fate_time"][k] = t
            state["fate_xv"][k] = xv            # 被抛出瞬间的日心状态矢量
            alive.discard(pid)
    events.clear()


def record_snapshot(sim, state, col_of, j, sim_order):
    sun = sim.particles[0]
    na = sim.N_active
    for i in range(na, sim.N):
        p = sim.particles[i]
        k = col_of[int(sim_order[i - na])]
        o = p.orbit(primary=sun)                # 显式用日心根数（默认是 Jacobi）
        state["elems"][k, j] = (o.a, o.e, o.inc, o.Omega, o.omega, o.M, o.d)
        state["xv"][k, j] = helio_xv(p, sun)
    # 大天体：存模拟坐标系（质心系）下的状态矢量。
    # 粒子质心系坐标 = 粒子日心坐标 + 太阳(第0个)质心系坐标
    for i in range(sim.N_active):
        p = sim.particles[i]
        state["bodies_xv"][i, j] = (p.x, p.y, p.z, p.vx, p.vy, p.vz)
    state["dE"][j] = (sim.energy() - float(state["E0"])) / abs(float(state["E0"]))


def save_npz_atomic(path, state, compressed):
    tmp = path[:-4] + ".tmp.npz"
    (np.savez_compressed if compressed else np.savez)(tmp, **state)
    os.replace(tmp, path)


def write_checkpoint(sim, state, files):
    tmp_sim = files["ckpt_sim"] + ".tmp"
    if os.path.exists(tmp_sim):
        os.remove(tmp_sim)
    sim.save_to_file(tmp_sim)
    os.replace(tmp_sim, files["ckpt_sim"])
    save_npz_atomic(files["ckpt_npz"], state, compressed=False)


def new_task_state(task, cfg, n_snap):
    tid, n = task["task_id"], task["n"]
    rng = np.random.default_rng([cfg["seed"], tid])   # 每个任务独立、可复现
    q = rng.uniform(task["q_min"], task["q_max"], n)
    a = 0.5 * (q + cfg["Q"])
    e = 1.0 - q / a
    inc = np.full(n, cfg["inc"])
    if cfg["random_angles"]:
        Om, om, M = rng.uniform(0.0, 2 * np.pi, size=(3, n))
    else:
        Om = om = M = np.zeros(n)
    pids = np.arange(task["first_pid"], task["first_pid"] + n)

    sim = rebound.Simulation(paths(cfg)["base"])
    sim.N_active = sim.N
    sim.testparticle_type = 0
    for k in range(n):
        sim.add(primary=sim.particles[0], a=a[k], e=e[k], inc=inc[k],
                Omega=Om[k], omega=om[k], M=M[k])
        set_label(sim.particles[-1], tp_label(pids[k]))

    state = dict(
        pid=pids, task_id=np.full(n, tid), region=np.full(n, task["region"], dtype="U4"),
        q0=q, a0=a, e0=e, inc0=inc, Omega0=Om, omega0=om, M0=M,
        t_start=np.array(sim.t),
        t_out=sim.t + cfg["dt_out"] * np.arange(1, n_snap + 1),
        elems=np.full((n, n_snap, len(FIELDS)), np.nan),
        xv=np.full((n, n_snap, len(FIELDS_XV)), np.nan),
        bodies_xv=np.full((len(body_labels(cfg)), n_snap, len(FIELDS_XV)), np.nan),
        dE=np.full(n_snap, np.nan),
        E0=np.array(sim.energy()),
        fate=np.zeros(n, dtype=np.int8),
        fate_time=np.full(n, np.nan),
        fate_xv=np.full((n, len(FIELDS_XV)), np.nan),
        snap_done=np.array(0),
        sim_order=pids.copy(),                  # 模拟中测试粒子的顺序（随粒子移除更新）
        config=np.array(config_json(cfg)),
    )
    return sim, state


def try_resume(cfg, files):
    if not (os.path.exists(files["ckpt_npz"]) and os.path.exists(files["ckpt_sim"])):
        return None, None
    try:
        with np.load(files["ckpt_npz"]) as d:
            state = {k: d[k].copy() for k in d.files}
        sim = rebound.Simulation(files["ckpt_sim"])
        j = int(state["snap_done"])
        if j == 0 or abs(sim.t - state["t_out"][j - 1]) > 2 * cfg["dt"] + 1e-6:
            return None, None                   # 两个文件不一致 → 从头算
        if "sim_order" not in state or len(state["sim_order"]) != sim.N - sim.N_active:
            return None, None
        return sim, state
    except Exception:
        return None, None


def integrate_task(task):
    cfg = task["cfg"]
    files = task_files(cfg, task["task_id"])
    n_snap = int(round(cfg["t_end"] / cfg["dt_out"]))
    sim, state = try_resume(cfg, files)
    resumed = sim is not None
    if not resumed:
        sim, state = new_task_state(task, cfg, n_snap)

    events = []
    configure(sim, cfg)

    col_of = {int(pid): k for k, pid in enumerate(state["pid"])}
    alive = {int(pid) for k, pid in enumerate(state["pid"]) if np.isnan(state["fate_time"][k])}
    sim_order = [int(x) for x in state["sim_order"]]
    if set(sim_order) != alive:
        raise RuntimeError("sim_order inconsistent with alive particles")
    exact = 1 if cfg["integrator"] == "ias15" else 0   # 辛积分器不强制落在整点，保持辛性

    for j in range(int(state["snap_done"]), n_snap):
        if not alive:
            break
        t_target = state["t_out"][j]
        while True:
            try:
                sim.integrate(t_target, exact_finish_time=exact)
                break
            except rebound.Escape:
                if remove_escapers(sim, events, cfg["exit_max_distance"], sim_order) == 0:
                    raise RuntimeError("Escape raised but no test particle beyond exit distance")
        apply_events(events, state, col_of, alive)
        if alive:
            record_snapshot(sim, state, col_of, j, sim_order)
        state["snap_done"] = np.array(j + 1)
        state["sim_order"] = np.array(sim_order, dtype=np.int64)
        if (j + 1) % cfg["ckpt_every"] == 0 and j + 1 < n_snap and alive:
            write_checkpoint(sim, state, files)

    # 存活粒子
    for pid in alive:
        k = col_of[pid]
        state["fate"][k] = FATE_SURVIVED
        state["fate_time"][k] = state["t_out"][-1]
        state["fate_xv"][k] = state["xv"][k, -1]

    save_npz_atomic(files["final"], state, compressed=True)
    for f in (files["ckpt_npz"], files["ckpt_sim"]):
        if os.path.exists(f):
            os.remove(f)
    return resumed


def run_task(task):
    """进程池入口：永远返回结果而不抛异常，一个任务出错不影响其它任务。"""
    tid, bodies = task["task_id"], task["cfg"]["bodies"]
    files = task_files(task["cfg"], tid)
    if os.path.exists(files["final"]):
        return bodies, tid, "skipped", 0.0, ""
    t0 = time.time()
    try:
        resumed = integrate_task(task)
        return bodies, tid, ("resumed+done" if resumed else "done"), time.time() - t0, ""
    except Exception:
        return bodies, tid, "ERROR", time.time() - t0, traceback.format_exc()


# =============================================================================
# 4. 合并结果
# =============================================================================
def merge(cfg, to_file=False):
    p = paths(cfg)
    lf = [p["log"]] if to_file else []
    pat = re.compile(r"task_(\d+)\.npz$")
    files = sorted(f for f in os.listdir(p["tasks"]) if pat.match(f))
    if not files:
        log(f"[{cfg['bodies']}] No finished task files to merge.", lf)
        return
    parts = []
    for f in files:
        with np.load(os.path.join(p["tasks"], f)) as d:
            parts.append({k: d[k] for k in d.files})
    configs = {str(d["config"]) for d in parts}
    if len(configs) != 1:
        sys.exit("Error: task files were produced with different settings; refusing to merge.")

    # 粒子维度的数组：沿第 0 维（粒子）拼接，再按 pid 排序
    keys_per_particle = ["pid", "task_id", "region", "q0", "a0", "e0", "inc0", "Omega0",
                         "omega0", "M0", "fate", "fate_time", "fate_xv", "elems", "xv"]
    merged = {k: np.concatenate([d[k] for d in parts]) for k in keys_per_particle}
    order = np.argsort(merged["pid"])
    merged = {k: v[order] for k, v in merged.items()}
    merged["t_out"] = parts[0]["t_out"]
    merged["t_start"] = parts[0]["t_start"]
    merged["fields"] = np.array(FIELDS)
    merged["fields_xv"] = np.array(FIELDS_XV)
    # 任务维度的数组：每个任务的行星轨迹各自独立（不同任务的行星会因混沌逐渐分叉）
    merged["task_ids"] = np.array([int(d["task_id"][0]) for d in parts])
    merged["bodies_xv"] = np.stack([d["bodies_xv"] for d in parts])   # (任务, 天体, 快照, 6)
    merged["dE"] = np.stack([d["dE"] for d in parts])                  # (任务, 快照)
    run_cfg = json.loads(str(parts[0]["config"]))      # 以文件中记录的参数为准
    merged["body_labels"] = np.array(run_cfg["body_labels"])
    merged["config"] = parts[0]["config"]
    save_npz_atomic(p["merged"], merged, compressed=True)

    n_exp = sum(run_cfg.get("n_per_region") or r[3] for r in REGIONS)
    log(f"[{cfg['bodies']}] Merged {len(merged['pid'])}/{n_exp} particles from {len(files)} tasks -> {p['merged']}", lf)
    for region in [r[0] for r in REGIONS]:
        m = merged["region"] == region
        if m.any():
            counts = {FATE_NAMES[c]: int(np.sum(merged["fate"][m] == c)) for c in FATE_NAMES}
            log(f"[{cfg['bodies']}]   {region}: {counts}", lf)


# =============================================================================
# 5. 主程序
# =============================================================================
def log(msg, log_files=()):
    """先写日志文件（只由主进程写入，不会冲突），再打印到终端。"""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    for lf in log_files:
        with open(lf, "a") as fh:
            fh.write(line + "\n")
    try:
        print(line, flush=True)
    except (BrokenPipeError, OSError):        # 终端断开时不让主进程崩掉
        pass


def fmt_dt(sec):
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    return f"{h}h {m}min" if h else f"{m}min {int(sec % 60)}s"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_json(cfg):
    """影响计算结果的全部参数（不含进程数、输出目录），存入每个 npz 的 "config" 字段。"""
    snap = {k: v for k, v in cfg.items() if k not in ("workers", "out_dir")}
    snap["body_labels"] = body_labels(cfg)
    snap["regions"] = REGIONS
    snap["start_date"] = START_DATE
    snap["rebound_version"] = rebound.__version__
    return json.dumps(snap, sort_keys=True)


def existing_task_files(cfg):
    pat = re.compile(r"task_\d+(\.partial)?\.npz$")
    d = paths(cfg)["tasks"]
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if pat.match(f)]


def check_config(cfg):
    """已有任务文件（含 checkpoint）的参数必须与本次完全相同，防止续算时混入不同设置的数据。"""
    current = json.loads(config_json(cfg))
    for f in existing_task_files(cfg):
        with np.load(f) as d:
            old = json.loads(str(d["config"])) if "config" in d.files else {}
        if old != current:
            diff = {k: (old.get(k), current.get(k)) for k in set(old) | set(current)
                    if old.get(k) != current.get(k)}
            sys.exit(f"Error: settings differ from existing file {f}:\n  {diff}\n"
                     f"Use another --out-dir or delete the old directory.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "merge", "build-base"])
    ap.add_argument("--bodies", nargs="+", default=[DEFAULTS["bodies"]], choices=sorted(BODY_SETS),
                    help="一个或多个天体组合，如 --bodies full giants（多个组合共用一个进程池）。"
                         "full: 太阳+8行星+5矮行星；giants: 太阳+4巨行星")
    ap.add_argument("--out-prefix", default="results",
                    help="输出目录前缀，目录为 <prefix>_<bodies>（默认 results_full、results_giants）")
    ap.add_argument("--out-dir", default=None, help="直接指定输出目录（只能配合单个天体组合）")
    ap.add_argument("--source-base", default=None,
                    help="build-base 时从已有的 base 文件挑选天体，不查询 Horizons")
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--particles-per-task", type=int, default=DEFAULTS["particles_per_task"])
    ap.add_argument("--n-per-region", type=int, default=None, help="覆盖每个区域的粒子数（试跑用）")
    ap.add_argument("--integrator", default=DEFAULTS["integrator"], choices=["trace", "mercurius", "ias15"])
    ap.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    ap.add_argument("--t-end", type=float, default=DEFAULTS["t_end"])
    ap.add_argument("--dt-out", type=float, default=DEFAULTS["dt_out"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return ap.parse_args()


def make_cfg(args, bodies):
    cfg = dict(DEFAULTS)
    cfg.update(bodies=bodies, out_dir=args.out_dir or f"{args.out_prefix}_{bodies}",
               workers=args.workers, particles_per_task=args.particles_per_task,
               n_per_region=args.n_per_region, integrator=args.integrator, dt=args.dt,
               t_end=args.t_end, dt_out=args.dt_out, seed=args.seed)
    return cfg


def prepare_set(cfg, args, full_base):
    """建立/检查一个天体组合的 base 文件和参数，返回 None；有问题时退出。"""
    p = paths(cfg)
    if args.command == "build-base" or not os.path.exists(p["base"]):
        if existing_task_files(cfg):
            sys.exit(f"Error: '{p['tasks']}' already contains task files; rebuilding the base "
                     f"simulation would mix different initial conditions. Use another output directory.")
        source = args.source_base
        # 非 full 组合：如果 full 组的 base 已存在，就从中挑选，保证初始状态一致且不必联网
        if source is None and cfg["bodies"] != "full" and full_base and os.path.exists(full_base):
            source = full_base
        build_base_simulation(cfg, source_base=source)
    if args.command == "build-base":
        return

    # base 文件中的天体必须与 --bodies 一致
    base_labels = read_labels_clean(p["base"])
    if base_labels != body_labels(cfg):
        sys.exit(f"Error: {p['base']} contains {base_labels}, but bodies '{cfg['bodies']}' "
                 f"expects {body_labels(cfg)}.")

    # base 文件的指纹也写进参数：base 被替换后，续算会被拒绝
    cfg["base_sha256"] = file_sha256(p["base"])
    check_config(cfg)


def main():
    args = parse_args()
    sets = list(dict.fromkeys(args.bodies))            # 去重，保持顺序
    if args.out_dir and len(sets) > 1:
        sys.exit("Error: --out-dir can only be used with a single --bodies value; use --out-prefix instead.")
    # full 放在最前：其他组合的 base 文件要从它挑选
    sets.sort(key=lambda b: b != "full")
    cfgs = {b: make_cfg(args, b) for b in sets}
    for cfg in cfgs.values():
        os.makedirs(paths(cfg)["tasks"], exist_ok=True)

    if args.command == "merge":
        for cfg in cfgs.values():
            merge(cfg)
        return

    full_base = paths(cfgs["full"])["base"] if "full" in cfgs else paths(make_cfg(args, "full"))["base"]
    if args.out_dir:
        full_base = args.source_base
    for cfg in cfgs.values():
        prepare_set(cfg, args, full_base)
    if args.command == "build-base":
        return

    # ---- 所有天体组合的任务放进同一个进程池 ----
    tasks, remaining, n_err, logs = [], {}, {}, {}
    for k, (b, cfg) in enumerate(cfgs.items()):
        ts = make_tasks(cfg)
        for t in ts:
            t["set_order"] = k
        tasks += ts
        remaining[b], n_err[b] = len(ts), 0
        logs[b] = [paths(cfg)["log"]]
        log("-" * 60, logs[b])
        log(f"[{b}] Start: {len(ts)} tasks, {sum(t['n'] for t in ts)} particles, "
            f"bodies={body_labels(cfg)}, shared pool of {cfg['workers']} workers with {sets}, "
            f"integrator={cfg['integrator']}, dt={cfg['dt']}, t_end={cfg['t_end']:.3g}, "
            f"dt_out={cfg['dt_out']:.3g}, rebound {rebound.__version__}", logs[b])

    # 长任务优先：先按区域（NK → UN → SU → JS），同一区域内 full 在前
    priority = {r: k for k, r in enumerate(SCHEDULE_ORDER)}
    tasks.sort(key=lambda t: (priority.get(t["region"], len(priority)), t["set_order"], t["task_id"]))
    workers = args.workers
    log(f"Pool: {len(tasks)} tasks in total ({', '.join(f'{b}: {remaining[b]}' for b in sets)}), "
        f"{workers} workers")

    t0 = time.time()
    n_done = 0
    # maxtasksperchild=1：每个任务用全新进程，避免长时间运行的内存累积
    with Pool(processes=workers, maxtasksperchild=1) as pool:
        for b, tid, status, dt, err in pool.imap_unordered(run_task, tasks, chunksize=1):
            n_done += 1
            remaining[b] -= 1
            if status == "ERROR":
                n_err[b] += 1
                log(f"[{b}] task {tid:05d} ERROR after {fmt_dt(dt)}:\n{err}", logs[b])
            elif status != "skipped":
                log(f"[{b}] task {tid:05d} {status} in {fmt_dt(dt)}  "
                    f"({n_done}/{len(tasks)}, elapsed {fmt_dt(time.time() - t0)})", logs[b])
            if remaining[b] == 0:
                # 这个天体组合的任务全部结束：立即合并（其他组合的任务继续在进程池中运行）
                log(f"[{b}] All tasks finished in {fmt_dt(time.time() - t0)}; errors: {n_err[b]}", logs[b])
                merge(cfgs[b], to_file=True)

    log(f"All body sets finished in {fmt_dt(time.time() - t0)}")


if __name__ == "__main__":
    main()