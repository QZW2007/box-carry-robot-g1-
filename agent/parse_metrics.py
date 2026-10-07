#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_metrics.py -- 把流水线各阶段的 stdout 日志解析成结构化指标。

设计目标（对应《项目掌握计划-v2.md》P1.2）:
    三种日志 -> 一个 dict，作为 diagnose() 的唯一输入层。

为什么需要它:
    流水线的失败信号散落在三段文本里(脚步脚本 print 的), Agent 要能"失败归因",
    必须先有机器可读的观测。**这个模块只做解析, 不做判断**, 判断在 diagnose.py。

已验证的真实日志格式来源(= 逐字抄的, 不是想象的):
    fit   : .workbuddy/batch_logs/20260922_165945_fit.log        (354 帧那条)
    world : .workbuddy/batch_logs/20260922_165945_world.log
    qa    : .workbuddy/batch_logs/qa_20260922_165945_boxpose_used.log

用法:
    from parse_metrics import parse_fit_log, parse_world_log, parse_qa_log, collect
    m = collect(fit_path=..., world_path=..., qa_path=...)
    print(m)   # 一个扁平 dict, 键见下面 SCHEMA

约定:
    - 解析失败不抛异常, 对应键缺省(或 None), 并在 'missing' 里列出。
      因为"某一行没出现"本身就是重要的失败信号(比如 fit 直接崩了)。
    - 所有数值统一成: 米 / 毫米 / 度 / 比例, 单位写进键名 (_mm, _deg, _m, _ratio)。
"""
from __future__ import annotations

import os
import re
import json
from typing import Optional

# ---------------------------------------------------------------------------
# 正则表: 每一行都对应真实日志里的一句
# 命名约定: (?P<name>...) 抽取后转 float
# ---------------------------------------------------------------------------

RE_TABLE = {

    # ---------------- 通用 ----------------
    "intr": re.compile(
        r"intr\s+(?P<wh>\d+x\d+)\s+fx=(?P<fx>[\d.]+)\s+fy=(?P<fy>[\d.]+)\s+"
        r"cx=(?P<cx>[\d.]+)\s+cy=(?P<cy>[\d.]+)"),
    "box_size": re.compile(
        r"cuboid fit ON: box (?P<w>[\d.]+) x (?P<h>[\d.]+) x (?P<d>[\d.]+) m, "
        r"half-extents \[(?P<hx>[\d.]+)\s+(?P<hy>[\d.]+)\s+(?P<hz>[\d.]+)\]"),

    # ---------------- fit 阶段 ----------------
    "frames_valid": re.compile(
        r"frames:\s+(?P<total>\d+)\s+valid 3D:\s+(?P<valid>\d+)\s+\((?P<pct>[\d.]+)%\)"),
    "longest_gap":  re.compile(r"longest gap:\s+(?P<n>\d+)\s+frames"),
    "fit_accepted": re.compile(
        r"cuboid fit \(silhouette \+ depth\):\s+(?P<ok>\d+)/(?P<total>\d+)\s+frames\s+\((?P<pct>[\d.]+)%\)"),
    # "ACCEPTED (354 f) -- silhouette IoU : median 0.914  p10 0.879  (occlusion lowers ..."
    "iou": re.compile(
        r"silhouette IoU\s*:\s*median\s+(?P<med>[\d.]+)\s+p10\s+(?P<p10>[\d.]+)"),
    # "ACCEPTED (354 f) -- depth residual : median 27 mm  p90 38 mm"
    "depth_res": re.compile(
        r"depth residual\s*:\s*median\s+(?P<med>[\d.]+)\s*mm\s+p90\s+(?P<p90>[\d.]+)\s*mm"),
    # "  ACCEPTED -- box z vs IMAGE up (NOT world up, camera is pitched): median 4.9 deg  p10 0.9 p90 15.6"
    "img_tilt": re.compile(
        r"box z vs IMAGE up.*?:\s*median\s+(?P<med>[\d.]+)\s+deg\s+p10\s+(?P<p10>[\d.]+)\s+p90\s+(?P<p90>[\d.]+)"),
    "bbox": re.compile(
        r"box bbox size \(metric, 20-80pct\):\s*W\s+(?P<w0>[\d.]+)\.\.(?P<w1>[\d.]+)\s*m\s+"
        r"H\s+(?P<h0>[\d.]+)\.\.(?P<h1>[\d.]+)\s*m"),

    # ---------------- world 阶段 ----------------
    # "rotation canonicalised over 354 measured frames (symmetry-equivalent solutions made continuous; 169 remaining >5 deg steps)"
    "canon": re.compile(
        r"rotation canonicalised over\s+(?P<n>\d+)\s+measured frames.*?;\s*"
        r"(?P<jumps>\d+)\s+remaining\s+>\s*5\s+deg steps\)"),
    "boxpose_cov": re.compile(
        r"box_pose:\s+(?P<cov>\d+)/(?P<total>\d+)\s+frames carry a 6-DoF measurement"),
    # "frame offset: bag[i+(-6)] <-> pkl[i]   (corr 0.621, 354 pkl / 354 bag frames)"
    "frame_offset": re.compile(
        r"frame offset:\s+bag\[i\+\((?P<off>-?\d+)\)\]\s*<->\s*pkl\[i\]\s+"
        r"\(corr\s+(?P<corr>[\d.]+),\s*(?P<npkl>\d+)\s+pkl\s*/\s*(?P<nbag>\d+)\s+bag frames\)"),
    # "iter 4: inliers 238/348  median 107 mm  p90 150 mm  s=0.716"
    # 只取最后一条一般迭代; 用 findall 后取最后一个
    "sim3_iter": re.compile(
        r"^iter\s+\d+:\s+inliers\s+(?P<inl>\d+)/(?P<tot>\d+)\s+median\s+(?P<med>\d+)\s*mm\s+"
        r"p90\s+(?P<p90>\d+)\s*mm\s+s=(?P<s>[\d.]+)", re.M),
    # "motion split: box leaves the floor at pkl f145 (height -1.093->-0.677 m, tol 4 mm), set down and still at f280"
    "motion_split": re.compile(
        r"motion split:\s+box leaves the floor at pkl f(?P<first>\d+)\s+"
        r"\(height\s+(?P<h0>-?[\d.]+)->(?P<h1>-?[\d.]+)\s*m,\s*tol\s+(?P<tol>[\d.]+)\s*mm\).*?"
        r"set down and still at f(?P<last>\d+)"),
    # "gravity from 212 parked frames: [+0.000, -0.999, -0.032] in camera coords (camera pitch vs -y_cam: 1.8 deg)"
    "gravity": re.compile(
        r"gravity from\s+(?P<n>\d+)\s+parked frames:\s*\[(?P<gx>-?[\d.]+),\s*(?P<gy>-?[\d.]+),\s*(?P<gz>-?[\d.]+)\]\s*"
        r"in camera coords \(camera pitch vs -y_cam:\s*(?P<pitch>[\d.]+)\s*deg\)"),
    "unconstrained_lean": re.compile(
        r"unconstrained fit leaned the up-axis\s+(?P<deg>[\d.]+)\s*deg off gravity"),
    # "axis check: cam-up vs world +z = 1.8 deg   toward-cam vs walk dir = 54.7 deg"
    "axis_check": re.compile(
        r"axis check:\s*cam-up vs world \+z\s*=\s*(?P<up>[\d.]+)\s*deg\s+"
        r"toward-cam vs walk dir\s*=\s*(?P<walk>[\d.]+)\s*deg"),
    # "scale camera->G1 world: 0.716  (G1 world is 1.40x real metres)"
    "scale": re.compile(
        r"scale camera->G1 world:\s*(?P<s>[\d.]+)\s*\(G1 world is\s*(?P<inv>[\d.]+)x real metres\)"),
    "inlier_gate": re.compile(
        r"inlier gate:\s*(?P<w>[\d.]+)\s*world m\s*=\s*(?P<r>[\d.]+)\s*real m"),
    "floor_z": re.compile(r"floor z \(G1 world\):\s*(?P<z>[-+]?[\d.]+)"),
    # "phase split: carry 136  ground 212  invalid 6"
    "phase_split": re.compile(
        r"phase split:\s*carry\s+(?P<carry>\d+)\s+ground\s+(?P<ground>\d+)\s+invalid\s+(?P<invalid>\d+)"),
    "world_frames": re.compile(
        r"6-DoF box pose in world coords:\s+(?P<n>\d+)\s+frames"),
    # "  on the FLOOR (pre-pickup): box CENTRE z median +0.319  spread 11 mm  (139 frames)"
    "floor_phase": re.compile(
        r"on the FLOOR \(pre-pickup\):\s*box CENTRE z median\s*[-+]?(?P<z>[\d.]+)\s+"
        r"spread\s+(?P<spread>\d+)\s*mm\s+\((?P<n>\d+)\s+frames\)"),
    "stool_phase": re.compile(
        r"on the STOOL \(post-drop\):\s*box CENTRE z median\s*[-+]?(?P<z>[\d.]+)\s+"
        r"spread\s+(?P<spread>\d+)\s*mm\s+\((?P<n>\d+)\s+frames\)"),
    # "floor->stool RISE of the box centre: +0.328 world m = +0.458 real m ; a 0.450 m stool needs +0.322 world m."
    "rise": re.compile(
        r"floor->stool RISE of the box centre:\s*[-+]?(?P<w>[\d.]+)\s*world m\s*=\s*"
        r"[-+]?(?P<r>[\d.]+)\s*real m\s*;\s*a\s+(?P<need_m>[\d.]+)\s*m stool needs\s*"
        r"[-+]?(?P<need_w>[\d.]+)\s*world m"),
    # "  box z-axis vs world up: median 175.8 deg  p90 179.2 deg"
    "world_zaxis": re.compile(
        r"box z-axis vs world up:\s*median\s+(?P<med>[\d.]+)\s*deg\s+p90\s+(?P<p90>[\d.]+)"),

    # ---------------- qa 阶段 ----------------
    # "loaded data/..._boxpose_used.npz : 354 frames, first_c=145 last_c=280, cov=348 carry=136"
    "qa_loaded": re.compile(
        r"loaded\s+\S+\s*:\s*(?P<n>\d+)\s+frames,\s*first_c=(?P<first>\d+)\s+last_c=(?P<last>\d+),\s*"
        r"cov=(?P<cov>\d+)\s+carry=(?P<carry>\d+)"),
    # "FLOOR parked   n=139 | visible rot step: med  0.000 deg  p90  0.000  max  0.390 | pos step: med   0.0 mm max   3.4 mm"
    "phase_stat": re.compile(
        r"(?P<phase>FLOOR parked|STOOL parked|CARRY)\s+n=\s*(?P<n>\d+)\s*\|\s*"
        r"visible rot step:\s*med\s+(?P<rmed>[\d.]+)\s*deg\s+p90\s+(?P<rp90>[\d.]+)\s*max\s+(?P<rmax>[\d.]+)\s*\|\s*"
        r"pos step:\s*med\s+(?P<pmed>[\d.]+)\s*mm\s+max\s+(?P<pmax>[\d.]+)\s*mm"),
    # "    WARN: parked pos jump 15.8 mm > 10 mm"
    "qa_warn_parkjump": re.compile(
        r"WARN:\s*parked pos jump\s+(?P<mm>[\d.]+)\s*mm\s*>\s*(?P<lim>[\d.]+)\s*mm"),
    # "[3] unmasked position-jump scan (> 50 mm)\n  lift/transition, expected f280->281    62.5 mm  (carry=1->0 cov=1->1)"
    "qa_jump": re.compile(
        r"^\s+(?P<label>[^()\n]+?)\s+f(?P<f0>\d+)->(?P<f1>\d+)\s+(?P<mm>[\d.]+)\s*mm\s+"
        r"\(carry=(?P<c0>\d)->(?P<c1>\d)\s+cov=(?P<v0>\d)->(?P<v1>\d)\)", re.M),
    "qa_result": re.compile(r"^RESULT:\s*(?P<res>PASS|FAIL)", re.M),
    "qa_clean": re.compile(r"unmasked position-jump scan \(> 50 mm\)\s*\n\s*none -- clean"),
    # fit 阶段的拒绝原因直方图, 例:
    #      308 x  IoU # < #
    #       22 x  mask too small
    #        8 x  no blob accepted
    # ★ 这是区分"掩码退化"与"先验尺寸错"的**唯一直接证据**:
    #   HSV 收窄 -> 大量 mask too small / no blob; box_size 错 -> 几乎全是 IoU < 门限。
    "reject_reason": re.compile(r"^\s+(?P<n>\d+)\s+x\s{2,}(?P<why>.+?)\s*$", re.M),
}


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _read(path: str) -> str:
    """读日志。容忍 utf-8 / utf-8-sig / utf-16 (PowerShell '>' 默认写 utf-16)。"""
    if not path or not os.path.isfile(path):
        return ""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "gbk"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 三个阶段各一个解析函数
# ---------------------------------------------------------------------------

def parse_fit_log(path: str) -> dict:
    """解析 `box_traj_from_depth.py --fit-cuboid` 的 stdout。"""
    txt = _read(path)
    out: dict = {"stage": "fit", "missing": [], "raw_len": len(txt)}

    if (m := RE_TABLE["intr"].search(txt)):
        out["intr_wh"] = m.group("wh")
        out["fx"] = _f(m.group("fx"))
    if (m := RE_TABLE["box_size"].search(txt)):
        out["box_w"] = _f(m.group("w"))
        out["box_h"] = _f(m.group("h"))
        out["box_d"] = _f(m.group("d"))
    if (m := RE_TABLE["frames_valid"].search(txt)):
        out["frames_total"] = _i(m.group("total"))
        out["frames_valid"] = _i(m.group("valid"))
        out["valid_ratio"] = round(_i(m.group("valid")) / max(1, _i(m.group("total"))), 4)
    else:
        out["missing"].append("frames_valid")
    if (m := RE_TABLE["longest_gap"].search(txt)):
        out["longest_gap_frames"] = _i(m.group("n"))
    if (m := RE_TABLE["fit_accepted"].search(txt)):
        out["fit_ok"] = _i(m.group("ok"))
        out["fit_total"] = _i(m.group("total"))
        out["fit_ok_ratio"] = round(_i(m.group("ok")) / max(1, _i(m.group("total"))), 4)
    else:
        # 完全没出现 "cuboid fit (silhouette + depth)" => 拟合阶段没跑起来
        out["missing"].append("fit_accepted")
    if (m := RE_TABLE["iou"].search(txt)):
        out["iou_median"] = _f(m.group("med"))
        out["iou_p10"] = _f(m.group("p10"))
    if (m := RE_TABLE["depth_res"].search(txt)):
        out["depth_res_med_mm"] = _f(m.group("med"))
        out["depth_res_p90_mm"] = _f(m.group("p90"))
    if (m := RE_TABLE["img_tilt"].search(txt)):
        out["img_tilt_med_deg"] = _f(m.group("med"))
        out["img_tilt_p90_deg"] = _f(m.group("p90"))
    if (m := RE_TABLE["bbox"].search(txt)):
        out["bbox_w_m"] = [_f(m.group("w0")), _f(m.group("w1"))]
        out["bbox_h_m"] = [_f(m.group("h0")), _f(m.group("h1"))]
    else:
        out["missing"].append("bbox")

    # ---- 拒绝原因直方图(只有拟合有失败帧时才出现) -------------------
    # 已知词表, 避免把 "40 frames..." 之类的进度行误吞进来。
    # 这是区分两类"拟合率低"的直接证据:
    #   mask too small / no blob  -> 掩码退化了(HSV 问题)
    #   纯 IoU < 门限             -> 轮廓对不上(尺寸/标定问题)
    reasons: dict = {}
    for m in RE_TABLE["reject_reason"].finditer(txt):
        why = m.group("why").strip()
        low = why.lower()
        if not any(k in low for k in ("iou", "mask", "blob", "depth",
                                      "not measured", "measurement")):
            continue
        # 数值归一化: "IoU 0.52 < 0.55" 与 "IoU # < #" 归成同一类
        key = re.sub(r"[-\d.]+", "#", why)
        reasons[key] = reasons.get(key, 0) + int(m.group("n"))
    if reasons:
        out["reject_reasons"] = reasons
        out["n_rejected"] = sum(reasons.values())
        # 掩码类拒绝(与几何无关, 说明像素根本没框住物体)
        out["n_reject_mask"] = sum(
            c for k, c in reasons.items()
            if "mask" in k.lower() or "blob" in k.lower())
        out["n_reject_iou"] = sum(
            c for k, c in reasons.items() if "iou" in k.lower())
        out["n_reject_depth"] = sum(
            c for k, c in reasons.items() if "depth" in k.lower())

        # ★ 掩码"倒挂": bbox 上界比下界还小, 说明形态学/连通域把掩码切碎了。
        #   实测: A6b(HSV 收窄) bbox_w_m=[0.284, 0.14] 倒挂;
        #         BADSIZE(尺寸错) bbox_w_m=[0.328, 0.52] 正常但 IoU 低。
        #   这是"掩码退化"区别于"先验尺寸错"的关键判据。
        bw, bh = out.get("bbox_w_m"), out.get("bbox_h_m")
        if bw and bh:
            inv_w = bw[1] < bw[0]
            inv_h = bh[1] < bh[0]
            out["bbox_inverted"] = bool(inv_w or inv_h)
            # 同时给出"掩码碎片化"的比例信号
            out["frac_reject_mask"] = round(
                out["n_reject_mask"] / max(1, out["n_rejected"]), 4)
    return out


def parse_world_log(path: str) -> dict:
    """解析 `align_box_to_world.py` 的 stdout。"""
    txt = _read(path)
    out: dict = {"stage": "world", "missing": [], "raw_len": len(txt)}

    if (m := RE_TABLE["canon"].search(txt)):
        out["canon_frames"] = _i(m.group("n"))
        out["canon_jumps_gt5deg"] = _i(m.group("jumps"))
    if (m := RE_TABLE["boxpose_cov"].search(txt)):
        out["boxpose_cov"] = _i(m.group("cov"))
        out["boxpose_total"] = _i(m.group("total"))
    if (m := RE_TABLE["frame_offset"].search(txt)):
        out["frame_offset"] = _i(m.group("off"))
        out["offset_corr"] = _f(m.group("corr"))
        out["offset_pkl_frames"] = _i(m.group("npkl"))
        out["offset_bag_frames"] = _i(m.group("nbag"))
    else:
        out["missing"].append("frame_offset")

    # sim3 迭代取最后一次(一般迭代 + 重力锁定迭代各有一串, 取全局最后一串)
    iters = list(RE_TABLE["sim3_iter"].finditer(txt))
    if iters:
        m = iters[-1]
        out["sim3_inliers"] = _i(m.group("inl"))
        out["sim3_total"] = _i(m.group("tot"))
        out["sim3_inlier_ratio"] = round(_i(m.group("inl")) / max(1, _i(m.group("tot"))), 4)
        out["sim3_resid_med_mm"] = _i(m.group("med"))
        out["sim3_resid_p90_mm"] = _i(m.group("p90"))
        out["sim3_s"] = _f(m.group("s"))

    if (m := RE_TABLE["motion_split"].search(txt)):
        out["ms_first_c"] = _i(m.group("first"))
        out["ms_last_c"] = _i(m.group("last"))
        out["ms_leave_h0"] = _f(m.group("h0"))
        out["ms_leave_h1"] = _f(m.group("h1"))
        out["ms_tol_mm"] = _f(m.group("tol"))
    else:
        out["missing"].append("motion_split")

    if (m := RE_TABLE["gravity"].search(txt)):
        out["grav_n"] = _i(m.group("n"))
        out["grav_cam"] = [_f(m.group("gx")), _f(m.group("gy")), _f(m.group("gz"))]
        out["grav_pitch_deg"] = _f(m.group("pitch"))
    if (m := RE_TABLE["unconstrained_lean"].search(txt)):
        out["lean_deg"] = _f(m.group("deg"))
    if (m := RE_TABLE["axis_check"].search(txt)):
        out["axis_up_deg"] = _f(m.group("up"))
        out["axis_walk_deg"] = _f(m.group("walk"))
    if (m := RE_TABLE["scale"].search(txt)):
        out["scale"] = _f(m.group("s"))
        out["scale_inv"] = _f(m.group("inv"))
    if (m := RE_TABLE["inlier_gate"].search(txt)):
        out["gate_world_m"] = _f(m.group("w"))
        out["gate_real_m"] = _f(m.group("r"))
    if (m := RE_TABLE["floor_z"].search(txt)):
        out["floor_z"] = _f(m.group("z"))
    if (m := RE_TABLE["phase_split"].search(txt)):
        out["n_carry"] = _i(m.group("carry"))
        out["n_ground"] = _i(m.group("ground"))
        out["n_invalid"] = _i(m.group("invalid"))
    if (m := RE_TABLE["world_frames"].search(txt)):
        out["world_frames"] = _i(m.group("n"))
    if (m := RE_TABLE["floor_phase"].search(txt)):
        out["floor_phase_z"] = _f(m.group("z"))
        out["floor_phase_spread_mm"] = _i(m.group("spread"))
        out["floor_phase_n"] = _i(m.group("n"))
    if (m := RE_TABLE["stool_phase"].search(txt)):
        out["stool_phase_z"] = _f(m.group("z"))
        out["stool_phase_spread_mm"] = _i(m.group("spread"))
        out["stool_phase_n"] = _i(m.group("n"))
    if (m := RE_TABLE["rise"].search(txt)):
        out["rise_world_m"] = _f(m.group("w"))
        out["rise_real_m"] = _f(m.group("r"))
        out["rise_needed_world_m"] = _f(m.group("need_w"))
    if (m := RE_TABLE["world_zaxis"].search(txt)):
        out["world_zaxis_med_deg"] = _f(m.group("med"))
        out["world_zaxis_p90_deg"] = _f(m.group("p90"))
    return out


def parse_qa_log(path: str) -> dict:
    """解析 `check_render_quality.py` 的 stdout。"""
    txt = _read(path)
    out: dict = {"stage": "qa", "missing": [], "raw_len": len(txt)}

    if (m := RE_TABLE["qa_loaded"].search(txt)):
        out["qa_frames"] = _i(m.group("n"))
        out["qa_first_c"] = _i(m.group("first"))
        out["qa_last_c"] = _i(m.group("last"))
        out["qa_cov"] = _i(m.group("cov"))
        out["qa_carry"] = _i(m.group("carry"))
    else:
        out["missing"].append("qa_loaded")

    phases: dict = {}
    for m in RE_TABLE["phase_stat"].finditer(txt):
        key = m.group("phase").split()[0]           # FLOOR / STOOL / CARRY
        phases[key] = {
            "n": _i(m.group("n")),
            "rot_med_deg": _f(m.group("rmed")),
            "rot_p90_deg": _f(m.group("rp90")),
            "rot_max_deg": _f(m.group("rmax")),
            "pos_med_mm": _f(m.group("pmed")),
            "pos_max_mm": _f(m.group("pmax")),
        }
    out["phases"] = phases
    if not phases:
        out["missing"].append("phases")

    out["warn_parked_pos_jump_mm"] = None
    if (m := RE_TABLE["qa_warn_parkjump"].search(txt)):
        out["warn_parked_pos_jump_mm"] = _f(m.group("mm"))

    jumps = []
    for m in RE_TABLE["qa_jump"].finditer(txt):
        jumps.append({
            "label": m.group("label").strip(" ,"),
            "f0": _i(m.group("f0")), "f1": _i(m.group("f1")),
            "mm": _f(m.group("mm")),
            "carry": (_i(m.group("c0")), _i(m.group("c1"))),
            "cov": (_i(m.group("v0")), _i(m.group("v1"))),
        })
    out["jumps"] = jumps
    out["clean"] = bool(RE_TABLE["qa_clean"].search(txt))

    if (m := RE_TABLE["qa_result"].search(txt)):
        out["result"] = m.group("res")
        out["pass"] = (m.group("res") == "PASS")
    else:
        # 没跑完/崩了 —— 对 diagnose 是重要信号
        out["result"] = None
        out["pass"] = None
        out["missing"].append("qa_result")

    # 派生量: 第一个失败帧号(面试要能一口报出)
    out["first_fail_frame"] = None
    if out.get("result") == "FAIL":
        cand = [j["f0"] for j in jumps if j["mm"] > 50.0]
        if cand:
            out["first_fail_frame"] = min(cand)
        elif out.get("warn_parked_pos_jump_mm") is not None:
            cand = [j["f0"] for j in jumps]
            out["first_fail_frame"] = min(cand) if cand else None
    return out


# ---------------------------------------------------------------------------
# 汇合
# ---------------------------------------------------------------------------

SCHEMA = """
扁平 dict, 键按阶段分组:

  fit   : frames_total, frames_valid, valid_ratio, longest_gap_frames,
          fit_ok, fit_total, fit_ok_ratio, iou_median, iou_p10,
          depth_res_med_mm, depth_res_p90_mm, img_tilt_med_deg, img_tilt_p90_deg,
          box_w,box_h,box_d, bbox_w_m, bbox_h_m, fx,
          reject_reasons{}, n_rejected, n_reject_mask, n_reject_iou,
          n_reject_depth, bbox_inverted, frac_reject_mask
  world : canon_frames, canon_jumps_gt5deg, boxpose_cov, boxpose_total,
          frame_offset, offset_corr, sim3_inliers, sim3_total, sim3_inlier_ratio,
          sim3_resid_med_mm, sim3_resid_p90_mm, sim3_s, scale, scale_inv,
          grav_n, grav_cam, grav_pitch_deg, lean_deg, axis_up_deg, axis_walk_deg,
          gate_world_m, gate_real_m, floor_z, n_carry, n_ground, n_invalid,
          world_frames, floor_phase_z, floor_phase_spread_mm, floor_phase_n,
          stool_phase_z, stool_phase_spread_mm, stool_phase_n,
          rise_world_m, rise_real_m, rise_needed_world_m,
          world_zaxis_med_deg, world_zaxis_p90_deg,
          ms_first_c, ms_last_c, ms_tol_mm
  qa    : qa_frames, qa_first_c, qa_last_c, qa_cov, qa_carry, phases{},
          warn_parked_pos_jump_mm, jumps[], clean, result, pass, first_fail_frame
  每个阶段都有: stage, missing[], raw_len
"""


def collect(fit_path: str = "", world_path: str = "",
            qa_path: str = "") -> dict:
    """把三个阶段日志一起解析成一个扁平 dict。缺哪个就少哪块。"""
    m: dict = {}
    if fit_path:
        m["fit"] = parse_fit_log(fit_path)
    if world_path:
        m["world"] = parse_world_log(world_path)
    if qa_path:
        m["qa"] = parse_qa_log(qa_path)

    flat: dict = {}
    for stage in ("fit", "world", "qa"):
        if stage in m:
            flat.update(m[stage])
    flat["_by_stage"] = m
    return flat


# ---------------------------------------------------------------------------
# 自测: 直接吃真实日志, 跑一遍看解出来的数字对不对
#   python parse_metrics.py <fit.log> [world.log] [qa.log]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        b = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        fix = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fixtures")
        qa = os.path.join(fix, "qa_20260922_165945.log")
        if not os.path.isfile(qa):
            qa = os.path.join(b, "qa_165945_batch_logs.log")
        args = [
            os.path.join(b, "165945_fit.log"),
            os.path.join(b, "165945_world.log"),
            qa,
        ]
    else:
        args = sys.argv[1:]
    got = collect(*args)
    got.pop("_by_stage", None)
    print(json.dumps(got, ensure_ascii=False, indent=2, default=str))

    # ---- 断言: 对 165945 这条片子的已知值做回归检查 ----
    # ★ 2026-10-07: QA 断言改用**交付产物**(数据/ 里的 v29 boxpose)重生成的
    #   fixtures 日志。原断言写 "result FAIL, first_fail_frame 280" 来自
    #   9/27 旧产物时代; 交付产品的 QA 是 FAIL 但首个失败帧是 **159**
    #   (拾取段 TELEPORT, 计划书附录B第8行的已知可接受项)。用错证据会让
    #   "测试通过"失去意义。详见 fixtures/README.md。
    exp = {
        "frames_total": 354, "fit_ok": 354, "iou_median": 0.914,
        "depth_res_med_mm": 27.0, "frame_offset": -6, "scale": 0.716,
        "sim3_s": 0.716, "ms_first_c": 145, "ms_last_c": 280,
        "n_carry": 136, "qa_first_c": 145, "qa_last_c": 280,
        "qa_carry": 136, "result": "FAIL", "first_fail_frame": 159,
        "canon_jumps_gt5deg": 169, "world_zaxis_med_deg": 175.8,
    }
    bad = [(k, got.get(k), v) for k, v in exp.items() if got.get(k) != v]
    print("\n[selftest]", "OK 全部命中" if not bad else f"FAIL {len(bad)} 项: {bad}")
    sys.exit(1 if bad else 0)
