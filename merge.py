# -*- coding: utf-8 -*-
"""
merge.py —— 合并任务结果（只用 numpy；不依赖 centaurs.py，可按需要随意修改）

读取 <总目录>/*/particles/*.npz 中已完成的任务（计算中途也可以运行，只合并已完成的部分）。
每个任务文件都自带说明：列名（elems_fields, xv_fields）、去向名称（fate_labels）、
天体名称（body_labels），以及任务名（task）。各数组的含义见 centaurs.py 开头的说明。

用法
-----
    python merge.py results_20261121             # 只生成跨构成的汇总表 summary.npz（很小）
    python merge.py results_20261121 --series    # 另外为每种构成生成完整的时间序列（很大）
    python merge.py results_20261121 --out 目录  # 输出目录，默认 <总目录>/merged

输出
-----
  summary.npz                     每个粒子一行（所有构成）
    body_set     (行,)            构成名（来自目录名 results_<构成>）
    pid, region  (行,)
    elems0, xv0  (行, 7) (行, 6)  初始根数、初始日心位置速度（= 任务文件 elems/xv 的第 0 列）
    fate, fate_time, fate_xv      去向、去向时刻、被抛出瞬间（或最后一个快照）的日心位置速度
    elems_fields, xv_fields, fate_labels
    body_labels__<构成>           各构成包含的天体
  <构成>_particles.npz  （--series）该构成全部粒子，按 pid 排序；数组与任务文件相同
  <构成>_bodies.npz     （--series）task_names (任务,)、bodies_xv (任务, 天体, 快照, 6)、dE (任务, 快照)
                                   每个任务的行星轨迹各自独立（不同任务的行星会因混沌逐渐分叉）

配对比较示例
-----
    s = np.load("results_20261121/merged/summary.npz")
    f, g = s["body_set"] == "full", s["body_set"] == "giants"
    common, i_f, i_g = np.intersect1d(s["pid"][f], s["pid"][g], return_indices=True)
    fate_full, fate_giants = s["fate"][f][i_f], s["fate"][g][i_g]
"""
import os
import sys
import glob
import argparse

import numpy as np


def load(path, keys=None):
    with np.load(path) as d:
        return {k: d[k] for k in (keys or d.files)}


def set_name_of(set_dir):
    name = os.path.basename(os.path.normpath(set_dir))
    return name.split("_", 1)[1] if name.startswith("results_") else name


def merge_set(set_dir, series, out_dir):
    """返回该构成的汇总部分；series=True 时另写完整时间序列。"""
    name = set_name_of(set_dir)
    pfiles = sorted(glob.glob(os.path.join(set_dir, "particles", "*.npz")))
    if not pfiles:
        return None
    small_keys = ["task", "pid", "region", "t", "fate", "fate_time", "fate_xv",
                  "body_labels", "elems_fields", "xv_fields", "fate_labels"]
    rows = {k: [] for k in ("pid", "region", "elems0", "xv0", "fate", "fate_time", "fate_xv")}
    full = {k: [] for k in ("task", "pid", "region", "elems", "xv", "fate", "fate_time", "fate_xv")} if series else None
    t_ref, meta = None, None
    for f in pfiles:
        d = load(f, small_keys + (["elems", "xv"]))
        if t_ref is None:
            t_ref = d["t"]
            meta = {k: d[k] for k in ("body_labels", "elems_fields", "xv_fields", "fate_labels")}
        elif not np.array_equal(d["t"], t_ref):
            sys.exit(f"Error: {f} has different snapshot times from the other tasks in {set_dir}")
        if not np.array_equal(d["body_labels"], meta["body_labels"]):
            sys.exit(f"Error: {f} has different bodies from the other tasks in {set_dir}")
        rows["pid"].append(d["pid"])
        rows["region"].append(d["region"])
        rows["elems0"].append(d["elems"][:, 0])
        rows["xv0"].append(d["xv"][:, 0])
        for k in ("fate", "fate_time", "fate_xv"):
            rows[k].append(d[k])
        if series:
            full["task"].append(np.full(len(d["pid"]), str(d["task"])))
            for k in ("pid", "region", "elems", "xv", "fate", "fate_time", "fate_xv"):
                full[k].append(d[k])
        del d
    part = {k: np.concatenate(v) for k, v in rows.items()}
    if len(np.unique(part["pid"])) != len(part["pid"]):
        sys.exit(f"Error: duplicate pids in {set_dir}")
    part["body_set"] = np.full(len(part["pid"]), name)

    print(f"[{name}] {len(pfiles)} tasks, {len(part['pid'])} particles, bodies: {', '.join(meta['body_labels'])}")
    for rg in np.unique(part["region"]):
        m = part["region"] == rg
        counts = ", ".join(f"{lb}: {int(np.sum(part['fate'][m] == c))}" for c, lb in enumerate(meta["fate_labels"]))
        print(f"    {rg}: {int(m.sum())} particles ({counts})")

    if series:
        # ---- 粒子：按 pid 排序 ----
        out = {k: np.concatenate(v) for k, v in full.items()}
        del full
        order = np.argsort(out["pid"])
        out = {k: v[order] for k, v in out.items()}
        out["t"] = t_ref
        out.update(meta)
        path = os.path.join(out_dir, f"{name}_particles.npz")
        np.savez_compressed(path, **out)
        del out
        # ---- 行星：任务维度；按任务名与粒子文件配对 ----
        tasks, bx, de = [], [], []
        for f in pfiles:
            task = os.path.basename(f)
            bf = os.path.join(set_dir, "bodies", task)
            if not os.path.exists(bf):
                print(f"    WARNING: {bf} missing; skipped in {name}_bodies.npz")
                continue
            d = load(bf)
            tasks.append(str(d["task"]))
            bx.append(d["bodies_xv"])
            de.append(d["dE"])
        np.savez_compressed(os.path.join(out_dir, f"{name}_bodies.npz"),
                            task_names=np.array(tasks), bodies_xv=np.stack(bx), dE=np.stack(de),
                            t=t_ref, body_labels=meta["body_labels"], xv_fields=meta["xv_fields"])
        print(f"    -> {name}_particles.npz, {name}_bodies.npz")
    return part, meta


def main():
    ap = argparse.ArgumentParser(description="合并任务结果")
    ap.add_argument("root", help="一次运行的总目录，例如 results_20261121")
    ap.add_argument("--series", action="store_true", help="同时为每种构成合并完整时间序列（文件很大）")
    ap.add_argument("--out", default=None, help="输出目录，默认 <root>/merged")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.root, "merged")
    os.makedirs(out_dir, exist_ok=True)
    set_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.root, "*", "particles")))
    if not set_dirs:
        sys.exit(f"No */particles directories under {args.root}")

    parts, summary = [], {}
    for sd in set_dirs:
        res = merge_set(sd, args.series, out_dir)
        if res is None:
            continue
        part, meta = res
        parts.append(part)
        summary[f"body_labels__{set_name_of(sd)}"] = meta["body_labels"]
        for k in ("elems_fields", "xv_fields", "fate_labels"):
            summary[k] = meta[k]
    if not parts:
        sys.exit("No finished tasks found.")
    for k in parts[0]:
        summary[k] = np.concatenate([p[k] for p in parts])
    path = os.path.join(out_dir, "summary.npz")
    np.savez_compressed(path, **summary)
    print(f"Summary: {len(summary['pid'])} rows -> {path}")


if __name__ == "__main__":
    main()
