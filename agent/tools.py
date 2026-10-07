#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools.py -- P1.1/P1.2 的 5 个工具：流水线的"可独立失败"的调用边界。

设计原则（《项目掌握计划-v2.md》P1.1）：
    一次调用 = 一个可独立失败的阶段，**不按代码函数切**。
    每个工具返回**机器可读的 JSON**(dict)，因为 diagnose() 要吃它。
    每个工具**幂等**：产物路径由调用方给定, 重复调用不污染状态, 不依赖任何全局量。
    每个工具带**它自己那一级的质量指标**, 不要只在最后一步才报错。

工具与底层脚本的对应（P1.2 "不要从零实现算法, 包既有脚本即可"）：
    calibrate_hsv   -> scripts/calib_hsv.py            (+ configs/boxes.json 回退)
    fit_box_6dof    -> scripts/box_traj_from_depth.py --fit-cuboid
    align_to_world  -> scripts/align_box_to_world.py
    render_and_qa   -> scripts/vis_robot_with_box.py + check_render_quality.py
    diagnose        -> 不调脚本, 直接调 diagnose.diagnose()

关于 stdout：本机 PowerShell 不回显子进程 stdout, 且受限沙箱下管道可能不可用。
所以这里**统一把子进程 stdout 重定向到日志文件**(不用 pipe), 再解析文件。
这与项目既有约定(.workbuddy/batch_logs/*.log)一致, 也让每次调用留下证据。

用法:
    from tools import fit_box_6dof, TOOLS
    r = fit_box_6dof(bag=..., hsv="4,75,79:24,255,255", out="agent/out/x_6dof.npz")
    print(r["ok"], r["metrics"])

自测(不跑重活, 只验证边界与日志解析):
    python tools.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
def _find_repo_root(start: str) -> str:
    """向上找到仓库根（含 scripts/ 与 setup.py 的那一层）。

    不写死 "../.." —— agent 代码可能在 <repo>/agent/ 也可能在
    <repo>/.workbuddy/agent/，写死层级会在移动后指向错误的目录。
    """
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "scripts")) and \
           os.path.isfile(os.path.join(d, "setup.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.abspath(os.path.join(start, ".."))


REPO = _find_repo_root(HERE)
PY = r"python"
QA = os.path.join(REPO, "skills",
                  "gmr-carry-video-pipeline", "scripts",
                  "check_render_quality.py")

sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# 统一执行层
# ---------------------------------------------------------------------------

def _rel(p: str) -> str:
    """绝对路径 -> 仓库相对路径(子进程 cwd=REPO, 日志里保持可读)。"""
    try:
        return os.path.relpath(p, REPO)
    except ValueError:
        return p


def _run(cmd: list, log_path: str, cwd: str = REPO,
         timeout: Optional[int] = None) -> dict:
    """跑一个阶段, stdout+stderr 写入 log_path(不用管道)。

    返回 {rc, log, secs}。**不抛异常** —— 阶段失败是一种结果, 不是崩溃,
    因为 diagnose() 恰恰需要"这一级失败了"这个信号。
    """
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("# cmd: %s\n" % " ".join(cmd))
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd=cwd, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            fh.write("\n# TIMEOUT after %ss\n" % timeout)
            rc = -9
        except FileNotFoundError as e:
            fh.write("\n# FileNotFoundError: %s\n" % e)
            rc = -2
    return {"rc": int(rc), "log": log_path, "secs": round(time.time() - t0, 1)}


def _sha(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 脚本指纹 —— 应对"代码版本漂移"型失败
# ---------------------------------------------------------------------------
# ★ 真实教训(2026-10-07): 用 run_v29.ps1 的**逐字参数**重跑交付片, 输入逐位相同、
#   参数逐字相同、渲染又是确定性的, 但 boxpose md5 仍是 b6b7108f 而非交付的
#   63473c15。最后靠 **mtime** 才发现: vis_robot_with_box.py 在 v29 渲染完成后
#   89 分钟被改过(手部穿模修复), 而那次改动会经 GRIP CLAMP 传到 box_pos
#   (carry 段 70 帧、中位 22.9 mm)。
#   结论: 每个阶段的产物都应带上**产生它的脚本指纹**, 否则"为什么复现不出来"
#   会变成一场考古。这里把指纹随工具输出一起返回。

_SCRIPTS = {
    "fit": "scripts/box_traj_from_depth.py",
    "cuboid_fit": "scripts/cuboid_fit.py",
    "world": "scripts/align_box_to_world.py",
    "render": "scripts/vis_robot_with_box.py",
    "qa": None,   # 解析自 QA 脚本路径
    "calib": "scripts/calib_hsv.py",
}


def fingerprints(*keys: str, extra: Optional[dict] = None) -> dict:
    """返回所选脚本的 {name: {sha256_16, mtime, path}}。

    用法: r["fingerprint"] = fingerprints("fit", "cuboid_fit")
    比较两次运行的指纹, 就能立刻区分"参数变了"与"引擎变了"。
    """
    out: dict = {}
    for k in keys:
        rel = _SCRIPTS.get(k)
        if rel is None:
            continue
        p = os.path.join(REPO, rel)
        out[k] = _fp(p)
    if extra:
        for k, p in extra.items():
            if not p:
                continue
            out[k] = _fp(p)
    return out


def _fp(path: str) -> dict:
    if not path or not isinstance(path, (str, bytes, os.PathLike)):
        return {"missing": True, "path": None}
    if not os.path.isfile(path):
        return {"missing": True, "path": _rel(str(path))}
    st = os.stat(path)
    return {"sha256_16": _sha(path),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(st.st_mtime)),
            "path": _rel(path)}


# ---------------------------------------------------------------------------
# 工具 1: calibrate_hsv
# ---------------------------------------------------------------------------

def calibrate_hsv(bag: str, out_dir: str = "agent/out",
                  hsv: Optional[str] = None, roi: Optional[str] = None,
                  frame: int = -1, timeout: int = 600) -> dict:
    """确定本场次的 HSV 区间。

    三种来源, 优先级最高的是显式传入:
      hsv 给定        -> 直接采用(不跑脚本), source="user-given"
      roi 给定        -> 跑 calib_hsv.py 交互式的无窗口版, source="measured"
      否则            -> 读 configs/boxes.json 的 session 匹配, source="config"

    失败信号(计划书 P1.1): 覆盖率 >40%(太宽) 或 <0.2%(太紧)。
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    res = {"tool": "calibrate_hsv", "bag": _rel(bag), "ok": False,
           "hsv": None, "source": None, "coverage_pct": None, "warnings": []}

    if hsv:
        res.update(hsv=hsv, source="user-given", ok=True)
        return res

    if roi:
        log = os.path.join(out_dir, "calibrate.log")
        cmd = [PY, "scripts/calib_hsv.py", "--bag", bag, "--roi", roi,
               "--no-show", "--out-preview",
               os.path.join(out_dir, "calib_preview.png")]
        if frame is not None:
            cmd += ["--frame", str(frame)]
        run = _run(cmd, log, timeout=timeout)
        res["secs"] = run["secs"]
        res["log"] = _rel(log)
        txt = _read(log)
        import re
        # "      use --hsv 4,75,79:24,255,255"
        m = re.search(r"use --hsv\s+(\S+)", txt)
        cm = re.search(r"coverage of the full frame:\s*([\d.]+)%", txt)
        if m:
            res["hsv"] = m.group(1)
            res["source"] = "measured(calib_hsv.py)"
        if cm:
            res["coverage_pct"] = float(cm.group(1))
            if res["coverage_pct"] > 40:
                res["warnings"].append("coverage>40pct: range too wide")
            elif res["coverage_pct"] < 0.2:
                res["warnings"].append("coverage<0.2pct: range too tight")
        res["ok"] = bool(res["hsv"])
        if not res["ok"]:
            res["error"] = "calib_hsv.py printed no --hsv range"
        return res

    # config 回退: 按 clip 前缀的 session 匹配(与 batch_pipeline 同一套规则)
    cfg = os.path.join(REPO, "configs", "boxes.json")
    sessions = _session_of(bag)
    try:
        with open(cfg, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:                                    # noqa: BLE001
        res["error"] = "cannot read %s: %s" % (_rel(cfg), e)
        return res
    for name, ent in (data or {}).items():
        if not isinstance(ent, dict):
            continue
        if ent.get("prefix") == sessions or ent.get("session") == _session_key(sessions):
            res.update(hsv=ent.get("hsv"), source="config:%s" % name, ok=True)
            return res
    res["error"] = ("no HSV for session %s in configs/boxes.json; "
                    "pass hsv= or roi=" % (sessions or "?"))
    return res


def _session_key(prefix: Optional[str]) -> Optional[str]:
    return prefix.split("_")[0] if prefix else None


def _session_of(bag: str) -> Optional[str]:
    """bag 路径 -> 规范前缀(复用项目的 clip_naming, 不自造轮子)。"""
    try:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        import clip_naming
        return clip_naming.canonical_prefix(bag) if not os.path.isabs(bag) \
            else clip_naming.canonical_prefix(os.path.join(REPO, bag))
    except Exception:                                         # noqa: BLE001
        import re
        m = re.search(r"(\d{8}_\d{6})", bag or "")
        return m.group(1) if m else None


# ---------------------------------------------------------------------------
# 工具 2: fit_box_6dof  ★ 结构化失败统计在这一级就产出
# ---------------------------------------------------------------------------

def fit_box_6dof(bag: str, hsv: str, out: str, box_size: Optional[str] = None,
                 preview: Optional[str] = None, min_iou: float = 0.55,
                 roi_pad: int = 280, min_area: int = 400,
                 max_frames: int = 0, timeout: int = 3600,
                 out_dir: str = "agent/out") -> dict:
    """深度 -> HSV 掩码 -> 跟踪 -> 6-DoF 拟合。产 npz + **失败统计**。

    返回值里的 metrics 直接来自 npz(而不是日志), 因为 npz 是权威产物;
    同时用 parse_metrics 解析日志, 补上拒绝原因直方图与掩码病灶信号。
    """
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log = os.path.join(os.path.abspath(out_dir), _tag(out) + "_fit.log")
    cmd = [PY, "-u", "scripts/box_traj_from_depth.py", "--bag", bag,
           "--fit-cuboid", "--hsv", hsv,
           "--roi-pad", str(roi_pad), "--min-area", str(min_area),
           "--min-iou", str(min_iou), "--out", out]
    if preview:
        cmd += ["--preview", os.path.abspath(preview)]
    if box_size:
        cmd += ["--box-size", str(box_size)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]

    run = _run(cmd, log, timeout=timeout)
    res = {"tool": "fit_box_6dof", "ok": False, "rc": run["rc"],
           "secs": run["secs"], "log": _rel(log), "out": _rel(out),
           "box_size": box_size or "0.30,0.30,0.45", "metrics": {},
           "fingerprint": fingerprints("fit", "cuboid_fit")}

    if os.path.isfile(out):
        res["metrics"].update(_npz_fit_metrics(out))
    try:
        from parse_metrics import parse_fit_log
        res["metrics"].update(parse_fit_log(log))
    except Exception as e:                                    # noqa: BLE001
        res["metrics"]["parse_error"] = str(e)
    res["metrics"].pop("stage", None)

    m = res["metrics"]
    res["ok"] = bool(run["rc"] == 0 and (m.get("fit_ok") or 0) > 0)
    # 这一级自己的质量指标(= diagnose 的输入)
    res["quality"] = {
        "fit_ok_ratio": m.get("fit_ok_ratio"),
        "iou_median": m.get("iou_median"),
        "iou_p10": m.get("iou_p10"),
        "depth_res_med_mm": m.get("depth_res_med_mm"),
        "depth_res_p90_mm": m.get("depth_res_p90_mm"),
    }
    return res


def _npz_fit_metrics(path: str) -> dict:
    """从 fit npz 直接算指标(权威来源, 不依赖日志格式)。"""
    import numpy as np
    d = np.load(path, allow_pickle=True)
    if "pose_valid" not in d.files:
        return {}
    pv = d["pose_valid"].astype(bool)
    iou = np.asarray(d["fit_iou"], dtype=float)
    dep = np.asarray(d["fit_depth_err_m"], dtype=float)
    ok = pv & np.isfinite(iou)
    dok = pv & np.isfinite(dep)
    out = {"frames_total": int(len(pv)), "fit_ok": int(pv.sum()),
           "fit_ok_ratio": round(float(pv.mean()), 4)}
    if ok.any():
        out["iou_median"] = round(float(np.median(iou[ok])), 3)
        out["iou_p10"] = round(float(np.percentile(iou[ok], 10)), 3)
    if dok.any():
        out["depth_res_med_mm"] = round(float(np.median(dep[dok]) * 1000), 1)
        out["depth_res_p90_mm"] = round(
            float(np.percentile(dep[dok], 90) * 1000), 1)
    import collections
    import re as _re
    if "fit_reason" in d.files:
        rs = [str(r) for r in d["fit_reason"] if str(r)]
        cnt = collections.Counter(_re.sub(r"[-\d.]+", "#", r) for r in rs)
        if cnt:
            out["reject_reasons"] = dict(cnt)
            out["n_rejected"] = int(sum(cnt.values()))
            low = {k.lower(): v for k, v in cnt.items()}
            out["n_reject_mask"] = sum(
                v for k, v in low.items() if "mask" in k or "blob" in k)
            out["frac_reject_mask"] = round(
                out["n_reject_mask"] / max(1, out["n_rejected"]), 4)
    return out


# ---------------------------------------------------------------------------
# 工具 3: align_to_world
# ---------------------------------------------------------------------------

def align_to_world(robot_motion: str, box_traj: str, out: str,
                   stool_h: float = 0.45, inlier_m: float = 0.22,
                   timeout: int = 1800,
                   out_dir: str = "agent/out") -> dict:
    """Sim3 对齐 + 相位分段。校验值: rise 偏差 / 轴上角 / carry 帧数。

    ★ `--box_traj` 与 `--box_pose` 都必须给且指向同一个 6dof npz
      (前者要位置, 后者要朝向); 只给 `--box_pose` 会静默回退到 legacy 默认值。
    """
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log = os.path.join(os.path.abspath(out_dir), _tag(out) + "_world.log")
    cmd = [PY, "scripts/align_box_to_world.py",
           "--robot_motion", robot_motion,
           "--box_traj", box_traj, "--box_pose", box_traj,
           "--out", out, "--stool_h", str(stool_h),
           "--inlier_m", str(inlier_m)]
    run = _run(cmd, log, timeout=timeout)
    res = {"tool": "align_to_world", "ok": False, "rc": run["rc"],
           "secs": run["secs"], "log": _rel(log), "out": _rel(out),
           "metrics": {}, "fingerprint": fingerprints("world", "cuboid_fit")}
    try:
        from parse_metrics import parse_world_log
        res["metrics"] = parse_world_log(log)
        res["metrics"].pop("stage", None)
        res["metrics"].pop("missing", None)
    except Exception as e:                                    # noqa: BLE001
        res["metrics"]["parse_error"] = str(e)
    res["ok"] = bool(run["rc"] == 0 and os.path.isfile(out))
    m = res["metrics"]
    rise_err = None
    if m.get("rise_real_m") is not None and stool_h:
        rise_err = round((m["rise_real_m"] - stool_h) * 1000, 1)   # mm
    res["quality"] = {
        "rise_real_m": m.get("rise_real_m"),
        "rise_err_mm_vs_stool": rise_err,
        "axis_up_deg": m.get("axis_up_deg"),
        "scale": m.get("scale"),
        "n_carry": m.get("n_carry"),
        "frame_offset": m.get("frame_offset"),
    }
    return res


# ---------------------------------------------------------------------------
# 工具 4: render_and_qa
# ---------------------------------------------------------------------------

def render_and_qa(robot_motion: str, box_pose_world: str, video: str,
                  dump_box_pose: str, qa_out: Optional[str] = None,
                  carry_rot_max: Optional[float] = None,
                  extra_args: Optional[list] = None,
                  timeout: int = 3600,
                  out_dir: str = "agent/out") -> dict:
    """渲染 + QA。返回 PASS/FAIL 与**首个失败帧号**。

    ★ carry_rot_max 不传时会**自动取 extra_args 里的 --quat_max_step_deg**。
      为什么必须联动(2026-10-07 实测的真实缺陷):
        QA 脚本的 CARRY_ROT_MAX 默认 3.5, 而渲染可以用
        --quat_max_step_deg 20(交付片 v29 就是)。两者不联动时, QA 会对
        **故意放宽限速的渲染**报 "carry rot max 12.28 > 3.5 deg slew cap",
        而那条片子在 SKILL 的验收里是合法的(限速是**为真实转动**放宽的)。
        实测 FULLRUN: 渲染 cap=20, QA 仍按 3.5 判 -> 多出一条无意义的 WARN。
      判据见 SKILL: "pass it as --carry_rot_max <cap>".
    """
    video = os.path.abspath(video)
    dump = os.path.abspath(dump_box_pose)
    os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
    tag = _tag(dump)
    rlog = os.path.join(os.path.abspath(out_dir), tag + "_render.log")
    cmd = [PY, "-u", "scripts/vis_robot_with_box.py",
           "--robot_motion", robot_motion,
           "--box_pose_world", box_pose_world,
           "--record_video", video, "--dump_box_pose", dump]
    if extra_args:
        cmd += list(extra_args)
    run = _run(cmd, rlog, timeout=timeout)

    # ---- QA 的 slew cap 跟随渲染的 cap(见 docstring 的实测依据) ----
    cap_src = "explicit"
    if carry_rot_max is None and extra_args:
        a = list(extra_args)
        for flag in ("--quat_max_step_deg",):
            if flag in a:
                i = a.index(flag)
                if i + 1 < len(a):
                    try:
                        carry_rot_max = float(a[i + 1])
                        cap_src = "from %s" % flag
                    except ValueError:
                        pass
                break

    qlog = os.path.abspath(qa_out or os.path.join(out_dir, tag + "_qa.log"))
    qcmd = [PY, QA, "--npz", dump]
    if carry_rot_max is not None:
        qcmd += ["--carry_rot_max", str(carry_rot_max)]
    qrun = _run(qcmd, qlog, timeout=900)

    res = {"tool": "render_and_qa", "ok": False, "rc": run["rc"],
           "render_secs": run["secs"], "qa_secs": qrun["secs"],
           "render_log": _rel(rlog), "qa_log": _rel(qlog),
           "video": _rel(video), "boxpose": _rel(dump),
           "video_sha": _sha(video), "metrics": {},
           "carry_rot_max": carry_rot_max, "carry_rot_max_src": cap_src,
           "fingerprint": fingerprints("render", extra={"qa": QA})}
    try:
        from parse_metrics import parse_qa_log
        res["metrics"] = parse_qa_log(qlog)
        res["metrics"].pop("stage", None)
        res["metrics"].pop("missing", None)
    except Exception as e:                                    # noqa: BLE001
        res["metrics"]["parse_error"] = str(e)
    m = res["metrics"]
    res["ok"] = bool(m.get("pass") is True)
    res["quality"] = {"result": m.get("result"),
                      "first_fail_frame": m.get("first_fail_frame"),
                      "clean": m.get("clean")}
    return res


# ---------------------------------------------------------------------------
# 工具 5: diagnose  (薄封装, 保证 5 个工具调用方式一致)
# ---------------------------------------------------------------------------

def diagnose_tool(metrics: Optional[dict] = None, *,
                  fit_log: str = "", world_log: str = "", qa_log: str = ""
                  ) -> dict:
    """把阶段指标映射成 (失败分类, 建议动作, 是否必须问人)。"""
    from diagnose import diagnose as _diag
    if metrics is None:
        from parse_metrics import collect
        metrics = collect(fit_log, world_log, qa_log)
    v = _diag(metrics)
    return {"tool": "diagnose", "cls": v.cls, "action": v.action,
            "ask": bool(v.ask), "confidence": v.confidence,
            "evidence": v.evidence, "note": v.note}


def _tag(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _read(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


TOOLS = {
    "calibrate_hsv": calibrate_hsv,
    "fit_box_6dof": fit_box_6dof,
    "align_to_world": align_to_world,
    "render_and_qa": render_and_qa,
    "diagnose": diagnose_tool,
}


# ---------------------------------------------------------------------------
# 自测: 只验证"边界与解析", 不跑重活
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("== tools.py selftest ==")
    fails = []

    # 1) 每个工具都有 docstring 与 JSON 化输出契约
    for name, fn in TOOLS.items():
        if not fn.__doc__:
            fails.append("%s has no docstring" % name)
    print("  5 tools registered:", ", ".join(sorted(TOOLS)))

    # 2) 对数轴上真实日志的解析: 三个已知案例
    cases = [
        ("healthy baseline", os.path.join(REPO, "agent", "fixtures",
                                         "165945_fit.log"), "ok"),
        ("A6b HSV narrowed", os.path.join(REPO, "agent", "fixtures",
                                          "A6b.log"), "mask_too_small"),
        ("BADSIZE wrong size", os.path.join(REPO, "agent", "fixtures",
                                            "BADSIZE.log"), "metric_mismatch"),
    ]
    from parse_metrics import parse_fit_log
    from diagnose import diagnose
    for tag, p, want in cases:
        if not os.path.isfile(p):
            fails.append("missing fixture %s" % p)
            continue
        v = diagnose(parse_fit_log(p))
        got = v.cls
        # healthy 基线是"全绿", 命中 ok 或信息性提示都算对
        good = (got == want) or (want == "ok" and got in
                                 ("ok", "zaxis_symmetry_branch"))
        print("  [%s] %-20s -> %-16s ask=%s" %
              ("PASS" if good else "FAIL", tag, got, v.ask))
        if not good:
            fails.append("%s: want %s got %s" % (tag, want, got))

    # 3) fit npz 直读指标(不依赖日志)
    #    包里随附 example/20260922_162859 一整套产物(见 README)
    npz = os.path.join(REPO, "example", "20260922_162859", "data",
                       "20260922_162859_6dof.npz")
    if os.path.isfile(npz):
        m = _npz_fit_metrics(npz)
        good = (m.get("fit_ok") == 418
                and abs(m.get("iou_median", 0) - 0.913) < 1e-3)
        print("  [%s] npz metrics (example 162859): fit_ok=%s iou_med=%s "
              "depth_med=%smm" %
              ("PASS" if good else "FAIL", m.get("fit_ok"),
               m.get("iou_median"), m.get("depth_res_med_mm")))
        if not good:
            fails.append("npz metrics wrong: %s" % m)
    else:
        fails.append("missing %s" % npz)

    # 4) 幂等契约: 工具签名里不得出现可变默认参数，且重复调用同一纯函数
    #    必须给出逐位相同的结果。
    #    ★ 原来这里写"往固定路径写两次文件再比 hash"，那**恒为真**（同一路径
    #      刚写完当然一样），是在测文件系统而不是测幂等 —— 而且还需要仓库写权限，
    #      克隆者放在只读位置就会误报失败。改成两条真有意义的检查。
    import inspect
    import hashlib as _hl
    bad_defaults = []
    for name, fn in TOOLS.items():
        for pname, prm in inspect.signature(fn).parameters.items():
            if isinstance(prm.default, (list, dict, set)):
                bad_defaults.append("%s(%s)" % (name, pname))
    if bad_defaults:
        fails.append("mutable default args: %s" % bad_defaults)
    print("  [%s] no mutable default args in the 5 tools"
          % ("PASS" if not bad_defaults else "FAIL"))

    # 纯函数重复调用一致性（幂等的可测部分）
    a1 = diagnose_tool({"fit_ok_ratio": 0.05, "bbox_inverted": True,
                        "frac_reject_mask": 0.1, "iou_median": None})
    a2 = diagnose_tool({"fit_ok_ratio": 0.05, "bbox_inverted": True,
                        "frac_reject_mask": 0.1, "iou_median": None})
    h1 = _hl.md5(json.dumps(a1, sort_keys=True, default=str).encode()).hexdigest()
    h2 = _hl.md5(json.dumps(a2, sort_keys=True, default=str).encode()).hexdigest()
    same = (h1 == h2)
    if not same:
        fails.append("diagnose not deterministic: %s vs %s" % (h1, h2))
    print("  [%s] diagnose is deterministic on the same metrics"
          % ("PASS" if same else "FAIL"))

    # 5) 脚本指纹: 必须能报出每个阶段的 sha256 与 mtime
    fp = fingerprints("fit", "cuboid_fit", "world", "render")
    need = {"fit", "cuboid_fit", "world", "render"}
    missing = [k for k in need if not (fp.get(k) or {}).get("sha256_16")]
    print("  [%s] fingerprints: %s" % ("PASS" if not missing else "FAIL",
                                       ", ".join("%s@%s" % (k, (fp.get(k) or {}).get("mtime", "?")[-8:])
                                                 for k in sorted(need))))
    if missing:
        fails.append("fingerprint missing for %s" % missing)

    print("\n=== %s ===" % ("ALL PASS" if not fails else
                            "FAIL %d: %s" % (len(fails), fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
