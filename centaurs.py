# -*- coding: utf-8 -*-
"""
centaurs.py —— 太阳系小天体长期演化模拟的计算部分（进程池版本）

本文件不含实验参数；参数全部在 main.py 中设置，由 main.py 调用 cli(cfg) 运行。
子进程用到的参数全部放在任务字典中显式传递，不读取任何由 main.py 设置的全局变量
（因此不依赖 multiprocessing 的启动方式：fork / spawn / forkserver 都能正确运行）。

一、任务如何生成
----------------
* BATCHES 每行 (区域, 任务数, 每任务粒子数, [构成...]) 展开为任务；同一组粒子在每种构成中各一个任务。
* 同一区域的粒子按 BATCHES 中各行的先后连续编号；pid = 区域 id × 1,000,000 + 区域内序号。
* 每个粒子的初始 q、Omega、omega、M 只由 (seed, 区域 id, 区域内序号) 和完整天体列表决定：
  依次抽取 q, Omega, omega, M；若初始位置在任一天体（BODIES 中太阳以外的全部天体，不论构成）的
  hill_factor × 希尔半径以内，就从同一随机数流抽取下一组。因此同一 pid 在各构成中初始条件完全相同。
* 任务名 <区域>_<区域内起点6位>_<粒子数3位>，例如 NK_000064_008。
* 启动顺序 = BATCHES 的行顺序，行内按任务交替各构成。

二、物理设定
------------
不做碰撞检测：所有天体都视为质点，粒子只有两种去向：
     0 = survived  存活到结束
     1 = ejected   距质心 r > exit_max_distance 即视为被抛出并移除
质点之间没有碰撞，粒子可能从行星"中心附近"穿过获得非物理的强烈弹弓，也可能以极小近日距掠过太阳，
需要在分析时由快照的 a, e（近日距 q）或与行星的距离自行甄别。

三、输出文件
------------
  results_<日期>/
    main.py                                开跑时复制的配置文件（内容变化时旧版本移入 config_history/）
    run_config.json                        运行配置（续算、追加批次时比对）
    run_log.txt                            总日志
    base_horizons_all.bin                  Horizons 下载的完整天体初始状态
    results_<构成>/
      base_solar_system.bin                该构成的初始天体状态
      run_log.txt                          该构成的日志（每个任务的完成情况、用时）
      particles/<任务名>.npz                粒子数据（此文件存在 = 任务完成）
      bodies/<任务名>.npz                   行星数据
      checkpoints/<任务名>.npz / .bin       运行中的 checkpoint，任务完成后删除

粒子文件（单位 AU, yr, rad；AU/yr）：
  task          任务名
  pid           (粒子,)              全局唯一编号
  region        (粒子,)              出发区域名
  t             (快照,)              快照时刻（年，自 J2000 起）；第 0 个是初始时刻
  elems         (粒子, 快照, 7)       日心根数，列名见 elems_fields；第 0 列是初始根数
  xv            (粒子, 快照, 6)       日心位置速度，列名见 xv_fields；第 0 列是初始状态
  fate          (粒子,)              去向编码，含义见 fate_labels（下标 = 编码）
  fate_time     (粒子,)              被抛出的时刻（精确到积分步）；存活粒子为最后一个快照时刻
  fate_xv       (粒子, 6)            被抛出瞬间的日心位置速度；存活粒子为最后一个快照的值
  body_labels   (天体,)              这次模拟包含的天体（即太阳系构成）
  elems_fields, xv_fields, fate_labels   列名、去向名称
  被抛出之后的快照为 NaN。
行星文件：
  task, t, body_labels, xv_fields
  bodies_xv     (天体, 快照, 6)       质心系位置速度；第 0 列是初始状态
  dE            (快照,)              相对能量误差（只反映行星积分；测试粒子质量为 0 不计入）
快照时刻 t 为名义时刻；辛积分器不强制落在整点，实际状态的时刻与 t 相差不超过一个积分步 dt。

REBOUND 5.2.2 注意：同一进程中"保存→释放→读取"文件后读回的粒子名称可能出错，
因此计算中不从文件读回名称（测试粒子用 sim_order 识别，名称检查放在干净的 spawn 进程中）。

依赖：rebound >= 4.4（推荐 5.x），numpy，astropy
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import re
import json
import shutil
import hashlib
import time
import argparse
import traceback
from datetime import datetime
from multiprocessing import Pool, get_context

import numpy as np
import rebound
from astropy.time import Time


ELEMS_FIELDS = ("a", "e", "inc", "Omega", "omega", "M", "r_helio")
XV_FIELDS = ("x", "y", "z", "vx", "vy", "vz")
FATE_SURVIVED, FATE_EJECTED = 0, 1
FATE_LABELS = ("survived", "ejected")
PID_BLOCK = 1_000_000
ANGLES = ("Omega", "omega", "M")       # 区域中按范围 (low, high) 均匀抽取的三个角
MAX_REDRAW = 100_000                    # 单个粒子最多重新抽取的次数

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TASK_FILE_RE = re.compile(r"^(?P<region>[A-Za-z0-9_-]+?)_(?P<start>\d{6})_(?P<n>\d{3})\.(?:npz|bin)$")


# =============================================================================
# REBOUND 4 / 5 兼容小工具（5.x 用 name 字符串，4.x 用 hash）
# =============================================================================
_HAS_NAME = hasattr(rebound.Particle, "name")


def set_label(p, label):
    if _HAS_NAME:
        p.name = label
    else:
        p.hash = label


def _read_labels(path, known_labels):
    sim = rebound.Simulation(path)
    if _HAS_NAME:
        return [sim.particles[i].name for i in range(sim.N)]
    table = {rebound.hash(s).value: s for s in known_labels}
    return [table.get(sim.particles[i].hash.value, "?") for i in range(sim.N)]


def read_labels_clean(path, known_labels):
    """在一个全新的进程（spawn）中读取 base 文件里的天体名称。
    REBOUND 5.2.2 在同一进程中"保存→释放→读取"后，读回的名称可能是错的，
    所以名称检查一律放在干净的进程里做。"""
    with get_context("spawn").Pool(1) as pool:
        return pool.apply(_read_labels, (path, list(known_labels)))


def integrator_name(sim):
    integ = sim.integrator
    return getattr(integ, "name", integ)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_json(obj):
    """规范化为 JSON 能表示的形式（tuple → list），用于比较和保存。"""
    return json.loads(json.dumps(obj))


# =============================================================================
# 配置检查
# =============================================================================
def validate_config(cfg):
    errs = []
    labels = [b[1] for b in cfg["BODIES"]]
    if len(set(labels)) != len(labels):
        errs.append("BODIES labels must be unique")
    for name, ls in cfg["BODY_SETS"].items():
        if not _NAME_RE.match(name):
            errs.append(f"body set name '{name}' must use letters, digits, '_' or '-'")
        unknown = [lb for lb in ls if lb not in labels]
        if unknown:
            errs.append(f"body set '{name}' contains unknown bodies {unknown}")
        elif list(ls) != [lb for lb in labels if lb in ls]:
            errs.append(f"body set '{name}': bodies must follow the order of BODIES")
        if not ls or ls[0] != "Sun":
            errs.append(f"body set '{name}' must start with 'Sun'")
    ids = [r["id"] for r in cfg["REGIONS"].values()]
    if len(set(ids)) != len(ids):
        errs.append(f"REGIONS ids must be unique: {ids}")
    for name, r in cfg["REGIONS"].items():
        if not _NAME_RE.match(name):
            errs.append(f"region name '{name}' must use letters, digits, '_' or '-'")
        missing = {"id", "q_min", "q_max", "Q", "inc", *ANGLES} - set(r)
        if missing:
            errs.append(f"region '{name}' missing {sorted(missing)}")
            continue
        if not (0 < r["q_min"] < r["q_max"] <= r["Q"]):
            errs.append(f"region '{name}': need 0 < q_min < q_max <= Q")
        for ang in ANGLES:
            v = r[ang]
            if not (isinstance(v, (tuple, list)) and len(v) == 2 and v[0] <= v[1]):
                errs.append(f"region '{name}': {ang} must be a range (low, high) with low <= high")
    for i, row in enumerate(cfg["BATCHES"]):
        if len(row) != 4:
            errs.append(f"BATCHES[{i}] must be (region, n_tasks, particles_per_task, [body sets])")
            continue
        region, n_tasks, ppt, sets = row
        if region not in cfg["REGIONS"]:
            errs.append(f"BATCHES[{i}]: unknown region '{region}'")
        if not (isinstance(n_tasks, int) and n_tasks > 0):
            errs.append(f"BATCHES[{i}]: n_tasks must be a positive integer")
        if not (isinstance(ppt, int) and 0 < ppt < 1000):
            errs.append(f"BATCHES[{i}]: particles_per_task must be an integer in 1..999")
        bad = [s for s in sets if s not in cfg["BODY_SETS"]]
        if bad or not sets or len(set(sets)) != len(sets):
            errs.append(f"BATCHES[{i}]: body sets {list(sets)} invalid (unknown: {bad})")
    sim = cfg["SIM"]
    if not (isinstance(sim.get("hill_factor"), (int, float)) and sim["hill_factor"] >= 0):
        errs.append("SIM hill_factor must be a number >= 0 (0 = no Hill-sphere check)")
    if sim["integrator"] not in ("trace", "mercurius", "ias15"):
        errs.append(f"SIM integrator '{sim['integrator']}' not supported")
    n_snap = sim["t_end"] / sim["dt_out"]
    if abs(n_snap - round(n_snap)) > 1e-9 or round(n_snap) < 1:
        errs.append("SIM t_end must be a positive integer multiple of dt_out")
    if errs:
        sys.exit("Configuration error:\n  " + "\n  ".join(errs))


# =============================================================================
# 路径
# =============================================================================
def root_paths(root):
    return dict(horizons=os.path.join(root, "base_horizons_all.bin"),
                log=os.path.join(root, "run_log.txt"),
                run_config=os.path.join(root, "run_config.json"),
                history=os.path.join(root, "config_history"))


def set_paths(root, prefix, bodies):
    d = os.path.join(root, f"{prefix}_{bodies}")
    return dict(dir=d,
                base=os.path.join(d, "base_solar_system.bin"),
                particles=os.path.join(d, "particles"),
                bodies=os.path.join(d, "bodies"),
                checkpoints=os.path.join(d, "checkpoints"),
                log=os.path.join(d, "run_log.txt"))


def task_name(region, start, n):
    return f"{region}_{start:06d}_{n:03d}"


def task_files(sp, name):
    return dict(particles=os.path.join(sp["particles"], name + ".npz"),
                bodies=os.path.join(sp["bodies"], name + ".npz"),
                ckpt_npz=os.path.join(sp["checkpoints"], name + ".npz"),
                ckpt_sim=os.path.join(sp["checkpoints"], name + ".bin"))


def make_set_dirs(sp):
    for k in ("particles", "bodies", "checkpoints"):
        os.makedirs(sp[k], exist_ok=True)


def scan_set(sp):
    """返回 {任务名: {"particles", "bodies", "ckpt"} 中存在的部分}"""
    out = {}
    for kind in ("particles", "bodies", "checkpoints"):
        if os.path.isdir(sp[kind]):
            for f in os.listdir(sp[kind]):
                if _TASK_FILE_RE.match(f):
                    out.setdefault(f.rsplit(".", 1)[0], set()).add("ckpt" if kind == "checkpoints" else kind)
    return out


# =============================================================================
# 1. 建立初始天体系统（主进程，只做一次）
# =============================================================================
def query_horizons(path, bodies, start_date):
    """按完整列表 BODIES 查询 Horizons，存为 path。"""
    labels = [b[1] for b in bodies]
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    print("Querying JPL Horizons ...")
    for query, label, mass in bodies:
        # 注意：必须传 date，否则 Horizons 返回的是"运行当下"的位置
        sim.add(query, date=start_date)
        part = sim.particles[-1]
        if mass is not None:
            part.m = mass
        set_label(part, label)
    jd = Time(start_date.replace(" ", "T"), format="isot", scale="utc").jd
    # 单位为 (AU, yr, Msun) 时时间单位就是"年"，不需要再乘 2π
    sim.t = (jd - 2451545.0) / 365.25
    sim.move_to_com()
    if os.path.exists(path):
        os.remove(path)
    sim.save_to_file(path)
    if read_labels_clean(path, labels) != labels:
        sys.exit(f"Error: verification of {path} failed")
    print(f"Saved {path}  (N = {sim.N}, t0 = {sim.t:.4f} yr since J2000)")


def derive_base(src, dst, wanted, all_labels, tag):
    """从完整的 base 文件中挑出 wanted 中的天体，移到新的质心系后存为 dst。"""
    have = read_labels_clean(src, all_labels)          # 名称在干净进程中读取
    missing = [lb for lb in wanted if lb not in have]
    if missing:
        sys.exit(f"Error: {src} does not contain {missing}")
    sim = rebound.Simulation(src)
    for i in range(sim.N - 1, -1, -1):                 # 按索引倒序删除，不读取粒子名称
        if have[i] not in wanted:
            sim.remove(i)
            del have[i]
    if have != list(wanted):
        sys.exit(f"Error: unexpected body order {have}, expected {list(wanted)}")
    # 天体组成变了，系统质心也变了，重新移到质心系（不改变各天体的日心状态）
    sim.move_to_com()
    if os.path.exists(dst):
        os.remove(dst)
    sim.save_to_file(dst)
    if read_labels_clean(dst, all_labels) != list(wanted):
        sys.exit(f"Error: verification of {dst} failed")
    print(f"[{tag}] base saved to {dst}  (N = {sim.N})")
    for i, label in enumerate(wanted):
        part = sim.particles[i]
        a = part.orbit(primary=sim.particles[0]).a if i else 0.0
        print(f"    {label:9s} m = {part.m:.4e} Msun   a_helio = {a:.4f} AU")
        if part.m == 0:
            print(f"    WARNING: {label} has zero mass!")


# =============================================================================
# 2. 任务分配
# =============================================================================
def expand_plan(cfg):
    """把 BATCHES 展开成任务列表（顺序即启动顺序）。
    每个任务 = (构成, 区域, 区域内起点, 粒子数)。同一区域的粒子按行的先后连续编号；
    同一行的同一组粒子在该行列出的每种构成中各生成一个任务（行内按任务交替各构成）。"""
    next_idx = {r: 0 for r in cfg["REGIONS"]}
    tasks, rows = [], []
    for b_i, (region, n_tasks, ppt, sets) in enumerate(cfg["BATCHES"]):
        first = next_idx[region]
        for k in range(n_tasks):
            start = next_idx[region]
            for bodies in sets:
                tasks.append(dict(bodies=bodies, region=region, start=start, n=ppt, batch=b_i,
                                  k_in_batch=k, name=task_name(region, start, ppt)))
            next_idx[region] += ppt
        if next_idx[region] > PID_BLOCK:
            sys.exit(f"Error: region '{region}' exceeds {PID_BLOCK} particles")
        rows.append(dict(batch=b_i, region=region, n_tasks=n_tasks, ppt=ppt, sets=list(sets),
                         first=first, last=next_idx[region] - 1))
    return tasks, rows


def hill_spheres(path):
    """从完整的 Horizons 文件中读出太阳以外每个天体的日心位置和希尔半径
    r_H = a (m / 3 M_sun)^(1/3)（a 为日心密切半长轴）。返回 [[x, y, z, r_H], ...]。
    只读位置和质量，不读粒子名称（顺序与 BODIES 相同）。"""
    sim = rebound.Simulation(path)
    sun = sim.particles[0]
    out = []
    for i in range(1, sim.N):
        p = sim.particles[i]
        a = p.orbit(primary=sun).a
        r_h = a * (p.m / (3.0 * sun.m)) ** (1.0 / 3.0)
        out.append([p.x - sun.x, p.y - sun.y, p.z - sun.z, r_h])
    return out


def helio_position(a, e, inc, Om, om, M):
    """由日心根数算日心位置（椭圆轨道；解开普勒方程）。"""
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E)
    cO, sO, co, so, ci, si = np.cos(Om), np.sin(Om), np.cos(om), np.sin(om), np.cos(inc), np.sin(inc)
    return np.array([(cO * co - sO * so * ci) * xp + (-cO * so - sO * co * ci) * yp,
                     (sO * co + cO * so * ci) * xp + (-sO * so + cO * co * ci) * yp,
                     (so * si) * xp + (co * si) * yp])


def initial_conditions(rp, start, n, seed, hill, hill_factor):
    """rp: 区域参数；hill: hill_spheres() 的结果。
    每个粒子用自己的随机数流 (seed, 区域 id, 区域内序号)，依次抽取 q, Omega, omega, M；
    若初始位置落在任一天体的 hill_factor × 希尔半径以内，就从同一个随机数流继续抽取下一组，
    直到满足为止。结果只由 (seed, 区域 id, 区域内序号) 和完整天体列表决定，与构成无关。
    返回 a, e, inc, Omega, omega, M 和每个粒子的重新抽取次数。"""
    hill = np.asarray(hill, dtype=float).reshape(-1, 4)
    centers, limits = hill[:, :3], hill_factor * hill[:, 3]
    out = np.empty((n, 5))                      # q, Omega, omega, M, 重抽次数
    inc = float(rp["inc"])
    for k in range(n):
        rng = np.random.default_rng([seed, rp["id"], start + k])
        for tries in range(MAX_REDRAW + 1):
            q = rng.uniform(rp["q_min"], rp["q_max"])
            Om, om, M = (rng.uniform(*rp[ang]) for ang in ANGLES)
            a = 0.5 * (q + rp["Q"])
            pos = helio_position(a, 1.0 - q / a, inc, Om, om, M)
            if hill_factor <= 0 or np.all(np.linalg.norm(centers - pos, axis=1) > limits):
                break
        else:
            raise RuntimeError(f"particle {start + k}: no position outside the Hill spheres "
                               f"after {MAX_REDRAW} draws")
        out[k] = (q, Om, om, M, tries)
    q = out[:, 0]
    a = 0.5 * (q + rp["Q"])
    e = 1.0 - q / a
    return (a, e, np.full(n, inc), out[:, 1].copy(), out[:, 2].copy(), out[:, 3].copy(),
            out[:, 4].astype(int))


# =============================================================================
# 3. 子进程：单个任务（只使用任务字典 t 中的参数）
# =============================================================================
def configure(sim, sc):
    if integrator_name(sim) != sc["integrator"]:
        sim.integrator = sc["integrator"]
    if sc["integrator"] in ("trace", "mercurius"):
        sim.dt = sc["dt"]
    sim.collision = "none"                    # 不做碰撞检测：所有天体视为质点
    sim.exit_max_distance = sc["exit_max_distance"]


def helio_xv(p, sun):
    return (p.x - sun.x, p.y - sun.y, p.z - sun.z, p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz)


# 测试粒子的识别方式：sim_order[k] = 模拟中第 (N_active + k) 个粒子的 pid。
# 不依赖 REBOUND 的粒子名称（REBOUND 5.2.2 读回的名称可能出错）。
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
    for i in range(na):                         # 大天体：模拟坐标系（质心系）
        p = sim.particles[i]
        state["bodies_xv"][i, j] = (p.x, p.y, p.z, p.vx, p.vy, p.vz)
    state["dE"][j] = (sim.energy() - float(state["E0"])) / abs(float(state["E0"]))


def save_npz_atomic(path, data, compressed):
    """先写临时文件（扩展名不是 .npz，不会被 merge.py 读到），完成后再改名。"""
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **data)
    os.replace(tmp, path)


# 结果文件中的内容；其余（E0、snap_done、sim_order、ckpt_key）只在 checkpoint 中
PARTICLE_KEYS = ("task", "pid", "region", "t", "elems", "xv", "fate", "fate_time", "fate_xv",
                 "body_labels", "elems_fields", "xv_fields", "fate_labels")
BODY_FILE_KEYS = ("task", "t", "bodies_xv", "dE", "body_labels", "xv_fields")


def new_task_state(t):
    sc, n = t["sim"], t["n"]
    n_snap = int(round(sc["t_end"] / sc["dt_out"]))
    a, e, inc, Om, om, M, redraws = initial_conditions(t["region_params"], t["start"], n, sc["seed"],
                                                       t["hill"], sc["hill_factor"])
    pids = t["first_pid"] + np.arange(n, dtype=np.int64)

    sim = rebound.Simulation(t["base"])
    sim.N_active = sim.N
    sim.testparticle_type = 0
    for k in range(n):
        sim.add(primary=sim.particles[0], a=a[k], e=e[k], inc=inc[k],
                Omega=Om[k], omega=om[k], M=M[k])
        set_label(sim.particles[-1], f"tp{int(pids[k])}")

    state = dict(
        task=np.array(t["name"]),
        pid=pids,
        region=np.full(n, t["region"], dtype="U16"),
        t=sim.t + sc["dt_out"] * np.arange(0, n_snap + 1),
        elems=np.full((n, n_snap + 1, len(ELEMS_FIELDS)), np.nan),
        xv=np.full((n, n_snap + 1, len(XV_FIELDS)), np.nan),
        bodies_xv=np.full((sim.N_active, n_snap + 1, len(XV_FIELDS)), np.nan),
        dE=np.full(n_snap + 1, np.nan),
        fate=np.zeros(n, dtype=np.int8),
        fate_time=np.full(n, np.nan),
        fate_xv=np.full((n, len(XV_FIELDS)), np.nan),
        body_labels=np.array(t["body_labels"]),
        elems_fields=np.array(ELEMS_FIELDS),
        xv_fields=np.array(XV_FIELDS),
        fate_labels=np.array(FATE_LABELS),
        # 以下只在 checkpoint 中
        E0=np.array(sim.energy()),
        snap_done=np.array(0),                  # 已记录的最后一个快照的下标
        sim_order=pids.copy(),                  # 模拟中测试粒子的顺序（随粒子移除更新）
        ckpt_key=np.array(t["key"]),
    )
    col_of = {int(pid): k for k, pid in enumerate(pids)}
    record_snapshot(sim, state, col_of, 0, list(pids))   # 第 0 列：初始状态
    return sim, state, int(redraws.sum())


def try_resume(t, files):
    if not (os.path.exists(files["ckpt_npz"]) and os.path.exists(files["ckpt_sim"])):
        return None, None
    try:
        with np.load(files["ckpt_npz"]) as d:
            state = {k: d[k].copy() for k in d.files}
    except Exception:
        return None, None
    if str(state.get("ckpt_key", "")) != t["key"]:
        raise RuntimeError(f"checkpoint {files['ckpt_npz']} was written with different settings")
    try:
        sim = rebound.Simulation(files["ckpt_sim"])
    except Exception:
        return None, None
    j = int(state["snap_done"])
    if j == 0 or abs(sim.t - state["t"][j]) > 2 * t["sim"]["dt"] + 1e-6:
        return None, None                       # 两个文件不一致 → 从头算
    if len(state["sim_order"]) != sim.N - sim.N_active:
        return None, None
    return sim, state


def write_checkpoint(sim, state, files):
    tmp_sim = files["ckpt_sim"] + ".part"
    if os.path.exists(tmp_sim):
        os.remove(tmp_sim)
    sim.save_to_file(tmp_sim)
    os.replace(tmp_sim, files["ckpt_sim"])
    save_npz_atomic(files["ckpt_npz"], state, compressed=False)


def integrate_task(t):
    """返回 (是否续算, 本次 sim.integrate 用时, 初始条件的重新抽取次数)。"""
    sc = t["sim"]
    files = t["files"]
    sim, state = try_resume(t, files)
    resumed = sim is not None
    redraws = 0
    if not resumed:
        sim, state, redraws = new_task_state(t)
    n_last = len(state["t"]) - 1

    events = []
    configure(sim, sc)

    col_of = {int(pid): k for k, pid in enumerate(state["pid"])}
    alive = {int(pid) for k, pid in enumerate(state["pid"]) if np.isnan(state["fate_time"][k])}
    sim_order = [int(x) for x in state["sim_order"]]
    if set(sim_order) != alive:
        raise RuntimeError("sim_order inconsistent with alive particles")
    exact = 1 if sc["integrator"] == "ias15" else 0   # 辛积分器不强制落在整点，保持辛性

    t_int = 0.0
    for j in range(int(state["snap_done"]) + 1, n_last + 1):
        if not alive:
            break
        t_target = state["t"][j]
        while True:
            t1 = time.perf_counter()
            try:
                sim.integrate(t_target, exact_finish_time=exact)
                t_int += time.perf_counter() - t1
                break
            except rebound.Escape:
                t_int += time.perf_counter() - t1
                if remove_escapers(sim, events, sc["exit_max_distance"], sim_order) == 0:
                    raise RuntimeError("Escape raised but no test particle beyond exit distance")
        apply_events(events, state, col_of, alive)
        if alive:
            record_snapshot(sim, state, col_of, j, sim_order)
        state["snap_done"] = np.array(j)
        state["sim_order"] = np.array(sim_order, dtype=np.int64)
        if j % t["ckpt_every"] == 0 and j < n_last and alive:
            write_checkpoint(sim, state, files)

    # 存活粒子
    for pid in alive:
        k = col_of[pid]
        state["fate"][k] = FATE_SURVIVED
        state["fate_time"][k] = state["t"][-1]
        state["fate_xv"][k] = state["xv"][k, -1]

    # 先写行星文件，再写粒子文件；粒子文件是"任务完成"的标志
    save_npz_atomic(files["bodies"], {k: state[k] for k in BODY_FILE_KEYS}, compressed=True)
    save_npz_atomic(files["particles"], {k: state[k] for k in PARTICLE_KEYS}, compressed=True)
    for f in (files["ckpt_npz"], files["ckpt_sim"]):
        if os.path.exists(f):
            os.remove(f)
    return resumed, t_int, redraws


def _worker_init():
    """子进程启动时离开终端的前台进程组：Ctrl-C 只发给主进程，由主进程统一结束进程池。
    （否则 SIGINT 会同时打断所有子进程，子进程可能在持有进程池内部锁时退出，导致主进程卡住。）"""
    if hasattr(os, "setpgrp"):
        os.setpgrp()


def run_task(t):
    """进程池入口：永远返回结果而不抛异常，一个任务出错不影响其它任务。"""
    files = t["files"]
    if os.path.exists(files["particles"]) and os.path.exists(files["bodies"]):
        return t["bodies"], t["name"], "skipped", 0.0, 0.0, 0, ""
    t0 = time.perf_counter()
    try:
        resumed, t_int, redraws = integrate_task(t)
        wall = time.perf_counter() - t0
        return t["bodies"], t["name"], ("resumed+done" if resumed else "done"), wall, t_int, redraws, ""
    except Exception:
        return t["bodies"], t["name"], "ERROR", time.perf_counter() - t0, 0.0, 0, traceback.format_exc()


# =============================================================================
# 4. 主进程：检查、日志
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


def run_config_record(cfg, sha):
    """影响结果、一次运行中不能改变的设置。"""
    return as_json(dict(
        start_date=cfg["START_DATE"], bodies=cfg["BODIES"], sim=cfg["SIM"],
        body_sets=cfg["BODY_SETS"], regions=cfg["REGIONS"], pid_block=PID_BLOCK,
        rebound_version=rebound.__version__,
        horizons_sha256=sha["horizons"],
        base_sha256={b: s for b, s in sha.items() if b != "horizons"},
    ))


def check_run_config(path, cfg, sha):
    """与总目录中的 run_config.json 比对。
    start_date、bodies、sim、rebound 版本、Horizons 文件必须完全相同；
    body_sets、regions、base 文件：已有的条目必须相同，允许新增。通过后写回（合并新增条目）。"""
    cur = run_config_record(cfg, sha)
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        errs = []
        for k in ("start_date", "bodies", "sim", "pid_block", "rebound_version", "horizons_sha256"):
            if old.get(k) != cur[k]:
                errs.append(f"{k}: {old.get(k)} -> {cur[k]}")
        for k in ("body_sets", "regions", "base_sha256"):
            for name, v in cur[k].items():
                if name in old.get(k, {}) and old[k][name] != v:
                    errs.append(f"{k}['{name}']: {old[k][name]} -> {v}")
            merged = dict(old.get(k, {}))
            merged.update(cur[k])
            cur[k] = merged
        if errs:
            sys.exit(f"Error: settings differ from {path}:\n  " + "\n  ".join(errs)
                     + "\nUse another --date/--root, or restore the previous settings in the config file.")
    cur["batches"] = as_json(cfg["BATCHES"])            # 仅供查阅，不比对
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(cur, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def check_existing(sp, bodies, planned):
    """已有的任务文件必须与本次计划一致：
    同名 → 同一个任务（全局设置已由 run_config.json 保证相同）；
    不同名但粒子范围重叠 → BATCHES 中已运行过的行被改动，拒绝运行。"""
    existing = scan_set(sp)
    by_name = {t["name"]: t for t in planned}
    errors, extra = [], []
    for name in sorted(existing):
        if name in by_name:
            continue
        m = _TASK_FILE_RE.match(name + ".npz")
        r, s, n = m.group("region"), int(m.group("start")), int(m.group("n"))
        clash = [t["name"] for t in planned if t["region"] == r and t["start"] < s + n and s < t["start"] + t["n"]]
        if clash:
            errors.append(f"{name}: particle range overlaps planned task {clash[0]}")
        else:
            extra.append(name)
    if errors:
        sys.exit(f"Error in {sp['dir']}:\n  " + "\n  ".join(errors[:20])
                 + ("\n  ..." if len(errors) > 20 else "")
                 + "\nAn existing BATCHES row was changed, removed or reordered within its region. "
                   "Restore it, or use another --date/--root.")
    if extra:
        print(f"[{bodies}] note: {len(extra)} existing task(s) are not in BATCHES; they are kept as they are.")


def copy_config(config_file, root):
    """把配置文件复制到总目录；内容有变化时，旧版本移入 config_history/。"""
    if not config_file or not os.path.exists(config_file):
        return
    dst = os.path.join(root, os.path.basename(config_file))
    if os.path.exists(dst):
        if file_sha256(dst) == file_sha256(config_file):
            return
        hist = root_paths(root)["history"]
        os.makedirs(hist, exist_ok=True)
        stamp = datetime.fromtimestamp(os.path.getmtime(dst)).strftime("%Y%m%d-%H%M%S")
        shutil.move(dst, os.path.join(hist, f"{stamp}_{os.path.basename(config_file)}"))
    shutil.copy2(config_file, dst)


def print_plan(cfg, root, rows, tasks):
    sc, prefix = cfg["SIM"], cfg["RUN"]["out_prefix"]
    n_snap = int(round(sc["t_end"] / sc["dt_out"]))
    print(f"Root: {root}   t_end = {sc['t_end']:.3g} yr, dt_out = {sc['dt_out']:.3g} yr "
          f"({n_snap} snapshots + initial), integrator = {sc['integrator']}, dt = {sc['dt']}, seed = {sc['seed']}")
    print("\nRegions (initial conditions):")
    for name, r in cfg["REGIONS"].items():
        print(f"  {name:6s} id={r['id']}  q = [{r['q_min']:.4f}, {r['q_max']:.4f}] AU  Q = {r['Q']}  "
              f"inc = {r['inc']}  " + "  ".join(f"{ang} = [{r[ang][0]:.4g}, {r[ang][1]:.4g}]" for ang in ANGLES))
    print(f"  Hill check: initial position outside {sc['hill_factor']} x Hill radius of every body in BODIES "
          f"(all planets and dwarf planets, regardless of body set)")
    print("\nBody sets:")
    for name, labels in cfg["BODY_SETS"].items():
        print(f"  {name:14s} {len(labels):2d} bodies: {', '.join(labels)}")
    existing = {b: scan_set(set_paths(root, prefix, b)) for b in cfg["BODY_SETS"]}
    print("\nBatches (in launch order):")
    print(f"  {'#':>2s} {'region':6s} {'tasks x ppt':>12s} {'particles':>9s}  {'index range':>15s}  "
          f"{'pid range':>17s}  body sets: done/running/total")
    for row in rows:
        n = row["n_tasks"] * row["ppt"]
        rid = cfg["REGIONS"][row["region"]]["id"]
        p0, p1 = rid * PID_BLOCK + row["first"], rid * PID_BLOCK + row["last"]
        st = []
        for b in row["sets"]:
            ts = [t for t in tasks if t["batch"] == row["batch"] and t["bodies"] == b]
            have = [existing[b].get(t["name"], set()) for t in ts]
            done = sum({"particles", "bodies"} <= h for h in have)
            running = sum("ckpt" in h for h in have)
            st.append(f"{b} {done}/{running}/{len(ts)}")
        print(f"  {row['batch']:2d} {row['region']:6s} {row['n_tasks']:5d} x {row['ppt']:<4d} {n:9d}  "
              f"{row['first']:6d} - {row['last']:<6d}  {p0:7d} - {p1:<7d}  {'; '.join(st)}")
    print("\nTotals per body set:")
    for b in cfg["BODY_SETS"]:
        ts = [t for t in tasks if t["bodies"] == b]
        if ts:
            print(f"  {b:14s} {len(ts):5d} tasks, {sum(t['n'] for t in ts):6d} particles")
    print(f"  all            {len(tasks):5d} tasks, {sum(t['n'] for t in tasks):6d} particle-runs")


# =============================================================================
# 5. 命令行入口（由 main.py 调用）
# =============================================================================
def cli(cfg, config_file=None):
    cfg = {k: cfg[k] for k in ("START_DATE", "BODIES", "BODY_SETS", "REGIONS", "BATCHES", "SIM", "RUN")}
    validate_config(cfg)
    run, prefix = cfg["RUN"], cfg["RUN"]["out_prefix"]
    all_labels = [b[1] for b in cfg["BODIES"]]

    ap = argparse.ArgumentParser(description="太阳系小天体长期演化模拟（参数在配置文件中设置）")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "plan", "build-base"])
    ap.add_argument("--date", default=None,
                    help="总目录名中的日期 YYYYMMDD，默认今天。续算、追加、plan 时必须指定原来的日期")
    ap.add_argument("--root", default=None, help="直接指定总目录（试跑用），优先于 --date")
    ap.add_argument("--only-bodies", nargs="+", default=None, choices=sorted(cfg["BODY_SETS"]),
                    help="只运行这些太阳系构成（不改变编号）")
    ap.add_argument("--only-regions", nargs="+", default=None, choices=sorted(cfg["REGIONS"]),
                    help="只运行这些区域（不改变编号）")
    ap.add_argument("--limit-tasks", type=int, default=None,
                    help="每个批次、每种构成只运行前 N 个任务（试跑用）")
    ap.add_argument("--workers", type=int, default=run["workers"])
    args = ap.parse_args()

    if args.root:
        root = args.root
    else:
        if args.date is None:
            args.date = datetime.now().strftime("%Y%m%d")
        elif not re.fullmatch(r"\d{8}", args.date):
            sys.exit(f"Error: --date must be YYYYMMDD, got '{args.date}'")
        root = f"{prefix}_{args.date}"

    all_tasks, rows = expand_plan(cfg)
    if args.command == "plan":
        print_plan(cfg, root, rows, all_tasks)
        return

    rp = root_paths(root)
    if not os.path.isdir(root):
        print(f"Creating new output directory {root}"
              + ("" if args.root else f"  (pass --date {args.date} to resume or append)"))
    os.makedirs(root, exist_ok=True)

    # ---- base 文件：Horizons 完整列表 → 各构成 ----
    if args.command == "build-base":
        busy = [b for b in cfg["BODY_SETS"] if scan_set(set_paths(root, prefix, b))]
        if busy or os.path.exists(rp["run_config"]):
            sys.exit(f"Error: {root} already has results; rebuilding the base would mix different "
                     f"initial conditions. Use another --date/--root.")
        query_horizons(rp["horizons"], cfg["BODIES"], cfg["START_DATE"])
        for b, labels in cfg["BODY_SETS"].items():
            sp = set_paths(root, prefix, b)
            make_set_dirs(sp)
            derive_base(rp["horizons"], sp["base"], labels, all_labels, b)
        return

    # ---- 选出本次要运行的任务（筛选不改变编号）----
    tasks = [t for t in all_tasks
             if (not args.only_bodies or t["bodies"] in args.only_bodies)
             and (not args.only_regions or t["region"] in args.only_regions)
             and (args.limit_tasks is None or t["k_in_batch"] < args.limit_tasks)]
    if not tasks:
        sys.exit("Nothing to run (check BATCHES / --only-* / --limit-tasks).")
    sets = [b for b in cfg["BODY_SETS"] if any(t["bodies"] == b for t in tasks)]

    if not os.path.exists(rp["horizons"]):
        if os.path.exists(rp["run_config"]) or any(os.path.exists(set_paths(root, prefix, b)["base"])
                                                    for b in cfg["BODY_SETS"]):
            sys.exit(f"Error: {rp['horizons']} is missing but {root} already has results; refusing to re-query.")
        query_horizons(rp["horizons"], cfg["BODIES"], cfg["START_DATE"])
    if read_labels_clean(rp["horizons"], all_labels) != all_labels:
        sys.exit(f"Error: {rp['horizons']} does not match BODIES in the config file.")
    sha = {"horizons": file_sha256(rp["horizons"])}
    hill = hill_spheres(rp["horizons"])         # 完整天体列表：各构成用同一份，初始条件才相同
    for b in sets:
        sp = set_paths(root, prefix, b)
        make_set_dirs(sp)
        if not os.path.exists(sp["base"]):
            if scan_set(sp):
                sys.exit(f"Error: {sp['base']} is missing but {sp['dir']} has task files.")
            derive_base(rp["horizons"], sp["base"], cfg["BODY_SETS"][b], all_labels, b)
        if read_labels_clean(sp["base"], all_labels) != list(cfg["BODY_SETS"][b]):
            sys.exit(f"Error: {sp['base']} does not match body set '{b}'.")
        sha[b] = file_sha256(sp["base"])

    # ---- 全局设置与已有结果比对，再逐个检查已有任务文件 ----
    check_run_config(rp["run_config"], cfg, sha)
    for b in sets:
        check_existing(set_paths(root, prefix, b), b, [t for t in all_tasks if t["bodies"] == b])
    copy_config(config_file, root)

    # ---- 组装任务：子进程需要的参数全部放进任务字典 ----
    for t in tasks:
        sp = set_paths(root, prefix, t["bodies"])
        rparams = dict(cfg["REGIONS"][t["region"]])
        t.update(sim=dict(cfg["SIM"]), ckpt_every=run["ckpt_every"], base=sp["base"],
                 files=task_files(sp, t["name"]), body_labels=list(cfg["BODY_SETS"][t["bodies"]]),
                 region_params=rparams, first_pid=rparams["id"] * PID_BLOCK + t["start"], hill=hill)
        key = as_json(dict(name=t["name"], bodies=t["bodies"], body_labels=t["body_labels"],
                           region_params=rparams, start=t["start"], n=t["n"], sim=t["sim"],
                           base_sha256=sha[t["bodies"]], horizons_sha256=sha["horizons"],
                           start_date=cfg["START_DATE"],
                           rebound_version=rebound.__version__))
        t["key"] = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

    sc = cfg["SIM"]
    remaining, n_err, logs = {}, {}, {}
    root_log = [rp["log"]]
    log("-" * 60, root_log)
    log(f"Start: root={root}, integrator={sc['integrator']}, dt={sc['dt']}, t_end={sc['t_end']:.3g}, "
        f"dt_out={sc['dt_out']:.3g}, seed={sc['seed']}, rebound {rebound.__version__}, "
        f"config={os.path.basename(config_file) if config_file else '?'}", root_log)
    for b in sets:
        ts = [t for t in tasks if t["bodies"] == b]
        remaining[b], n_err[b] = len(ts), 0
        logs[b] = [set_paths(root, prefix, b)["log"]]
        log("-" * 60, logs[b])
        per_region = {r: sum(t["n"] for t in ts if t["region"] == r) for r in cfg["REGIONS"]}
        log(f"[{b}] Start: {len(ts)} tasks, {sum(t['n'] for t in ts)} particles "
            f"({', '.join(f'{r}: {n}' for r, n in per_region.items() if n)}), "
            f"bodies={cfg['BODY_SETS'][b]}", logs[b] + root_log)
    log(f"Pool: {len(tasks)} tasks in total ({', '.join(f'{b}: {remaining[b]}' for b in sets)}), "
        f"{args.workers} workers, launch order as in BATCHES", root_log)

    t0 = time.time()
    n_done = 0
    # maxtasksperchild=1：每个任务用全新进程，避免长时间运行的内存累积
    pool = Pool(processes=args.workers, maxtasksperchild=1, initializer=_worker_init)
    try:
        for b, name, status, wall, t_int, redraws, err in pool.imap_unordered(run_task, tasks, chunksize=1):
            n_done += 1
            remaining[b] -= 1
            if status == "ERROR":
                n_err[b] += 1
                log(f"[{b}] task {name} ERROR after {fmt_dt(wall)}:\n{err}", logs[b])
            elif status != "skipped":
                frac = 100.0 * t_int / wall if wall > 0 else 0.0
                rd = f", Hill redraws {redraws}" if redraws else ""
                log(f"[{b}] task {name} {status} in {fmt_dt(wall)} (integrate {frac:.2f}%{rd})  "
                    f"({n_done}/{len(tasks)}, elapsed {fmt_dt(time.time() - t0)})", logs[b])
            if remaining[b] == 0:
                log(f"[{b}] All tasks finished in {fmt_dt(time.time() - t0)}; errors: {n_err[b]}",
                    logs[b] + root_log)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        log(f"Interrupted after {fmt_dt(time.time() - t0)} ({n_done}/{len(tasks)} tasks returned). "
            f"Checkpoints are kept; run the same command again to resume.",
            root_log + [lf for b in sets for lf in logs[b]])
        sys.exit(130)
    pool.close()
    pool.join()

    log(f"All body sets ({', '.join(sets)}) finished in {fmt_dt(time.time() - t0)}; "
        f"errors: {sum(n_err.values())}. Run `python merge.py {root}` to merge the results.",
        root_log + [lf for b in sets for lf in logs[b]])