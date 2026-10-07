#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_agent.py -- P1.4 的编排器：把 5 个工具按 calibrate→fit→align→render 串起来,
并把《项目掌握计划-v2.md》P1.3 的**三条铁律**落成可检查的代码。

    铁律 1  ask=True 的分类**绝不自动重试**
             —— 知道什么时候停下来问人, 是 Agent 成熟度的标志。
    铁律 2  重试必须**改变一件具体的事**(换 HSV / 加密初始器 / 重搜 offset),
             不是"再跑一次"。带随机种子的重跑不算重试。
    铁律 3  重试上限 **2 次**, 之后升级为 ask=True。

与 diagnose.run_with_retry 的关系:
    后者是"规则层"的演示(纯函数, 不碰流水线)。这里补上**真正执行**的部分:
    把 action 字符串翻译成对某个工具的**具体参数改动**, 并留下每次尝试的证据。
    两者共用 RETRYABLE / MAX_RETRIES, 不另立一套常量。

用法:
    # 1) 演示: 用已知的 A6b 故障日志离线跑(不启动重活), 看归因与重试决策
    python run_agent.py --replay agent/fixtures/A6b.log

    # 2) 真跑一条 clip(需要 bag 与 g1.pkl; 很慢)
    python run_agent.py --bag D:\\bag\\rgbd_20260922_165945\\realsense.bag \\
        --robot data/20260922_165945_g1.pkl --prefix 20260922_165945 \\
        --hsv 4,75,79:24,255,255 --box-size 0.30,0.30,0.45

    # 3) 故意用错 box_size, 验证"识别并停下问人"
    python run_agent.py --bag ... --robot ... --prefix ... --box-size 0.20,0.20,0.30

    # 4) 故意用错 HSV, 验证"自动放宽 HSV 并跑通"
    python run_agent.py --bag ... --robot ... --prefix ... --hsv 9,165,167:19,255,255
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
sys.path.insert(0, HERE)

from diagnose import RETRYABLE, MAX_RETRIES, diagnose      # noqa: E402
from parse_metrics import parse_fit_log, collect           # noqa: E402


# ---------------------------------------------------------------------------
# 渲染参数: 交付片 v29 的整套(.workbuddy/run_v29.ps1 逐字抄录)
# ---------------------------------------------------------------------------
# ★ 为什么要内置这一套: 渲染器的**默认值不等于交付参数**。
#   实测(2026-10-07): 只传最小四件套时, GRIP CLAMP 漂移上限 450->50 mm,
#   QA 首个失败帧 f159-164 -> f280-281, 最终判定 qa_known_acceptable ->
#   qa_fail_review。也就是说"能跑通"与"复现交付片"是两件事。
#   这里把交付用的整套作为 --full 的默认, 让二者合一。
#   要跑裸默认渲染: render_extra=[]; 要改单个值: 自己给完整 render_extra。
RENDER_TUNED = [
    "--box_size", "0.30", "0.30", "0.45",
    "--box_offset", "0.12", "0.0", "0.05",
    "--phase_lift", "85", "--phase_carry", "100",
    "--chair_size", "0.32", "0.32", "0.45",
    "--chair_offset", "0.476", "0.902",
    "--pose_z", "rest",
    "--box_sym_snap", "continuity",
    "--carry_tilt_anchor_deg", "5", "--carry_yaw_anchor_deg", "0",
    "--meas_median", "3", "--meas_smooth_quat", "5",
    "--body_yaw_monotone", "--quat_max_step_deg", "20",
    "--place_root_clear_m", "0.27", "--place_waist_lift_deg", "20",
    "--carry_grip_slack", "0.45", "--carry_grip_slack_walk", "0.02",
    "--carry_grip_slack_walk_win", "150", "165",
    "--mirror_left_end", "176",
    "--hand_push_out_mm", "30",
    "--carry_abduct_deg", "25", "--carry_abduct_win", "148", "195",
    "--hand_roll_deg", "0",
    "--place_bow_deg", "15", "--place_bow_win", "280", "316",
]
# 本轮实测: 渲染 cap=20 -> QA 的 --carry_rot_max 必须跟着 20, 否则误报
# "carry rot max 6.22 > 3.5 deg slew cap"(见 tools.render_and_qa 的 docstring)。
# 这条联动现在由 render_and_qa 自动完成, 不需要在这里重复传。


# ---------------------------------------------------------------------------
# 铁律 2 的落地: action 字符串 -> 一次**具体**的参数改动
# ---------------------------------------------------------------------------

def _relax_hsv(hsv: str, s_drop: int = 20, v_drop: int = 20,
               hue_margin: int = 10) -> str:
    """放宽 HSV —— 计划书 P1.3 的 relax_hsv(-20 on S/V floor)。

    ★ 这里有一课是**真跑出来的**, 不是推的: 只降 S/V 下限**修不好**本例。
      实测(165945, 完整 354 帧):
        A6b 故障  4,120,130:**14**       -> fit_ok 4.5%   bbox_w_m=[0.284,0.144] 倒挂
        只降 S/V  4,100,110:**14**       -> fit_ok 8%     bbox_w_m=[0.285,0.144] **仍倒挂**
        只放色相  4,120,130:**24**       -> fit_ok **99%** bbox_w_m=[0.298,0.499]
        两者都做  4,100,110:**24**       -> fit_ok **100%** IoU 0.898 bbox=[0.316,0.508]
      也就是说: 故障的主因是**色相上限被砍窄**(24->14 丢掉了一半橙色像素),
      S/V 下限只是次要因素。只改一手 = 违反了铁律 2 的精神
      ("改变一件具体的事"必须改到**要害**那件事, 而不是随便改一件)。
      所以这里**三手同时做**: 放宽 S/V 下限、色相上限补余量、色相下限也往外放
      (下限同样可能被错误收紧, 如极端案例 9,...,19 —— 真值下限是 4)。
      全部只做**单向放宽**, 不会把区间收窄。
    """
    try:
        lo, hi = hsv.split(":")
        lp = [int(x) for x in lo.split(",")]
        hp = [int(x) for x in hi.split(",")]
        h0 = lp[0] if lp else 0
        s = lp[1] if len(lp) > 1 else 0
        v = lp[2] if len(lp) > 2 else 0
        h1 = hp[0] if hp else 179

        h0b = max(0, h0 - hue_margin)          # 色相下限往外(不收紧)
        h1b = min(179, h1 + hue_margin)        # 色相上限往外(不收紧)
        s2 = max(0, s - s_drop)
        v2 = max(0, v - v_drop)

        lo2 = [str(h0b), str(s2), str(v2)]
        hi2 = [str(h1b)] + [str(x) for x in hp[1:]]
        return "%s:%s" % (",".join(lo2), ",".join(hi2))
    except Exception:                                          # noqa: BLE001
        return hsv


def apply_action(action: str, state: dict) -> dict:
    """把 diagnose 给的 action 翻成对 state 的具体改动。

    返回 {changed: bool, what: str, params: {...}}。
    **changed=False 时调用方必须停止重试** —— 因为"没改变任何东西的重试"
    违反铁律 2, 只是烧时间。
    """
    a = (action or "").lower()
    p = dict(state.get("params", {}))
    out = {"changed": False, "what": "", "params": p}

    if "relax_hsv" in a or "re-calibrate" in a or "hsv" in a:
        old = p.get("hsv")
        new = _relax_hsv(old) if old else old
        if new and new != old:
            p["hsv"] = new
            out.update(changed=True, what="hsv %s -> %s (S/V floor -20)" % (old, new))
    elif "densify_tilt" in a:
        # 初始器加密: -25/0/25 -> -40..40 step 10 是 cuboid_fit 里那 36 个起点的
        # tilt 维; 这里用环境变量传给 solve 的调用方(见 tools.fit_box_6dof 注)。
        p["tilt_step_deg"] = 10.0
        p["tilt_span_deg"] = 40.0
        out.update(changed=True, what="tilt initialisers densified to +-40 step 10")
    elif "frame_offset" in a:
        p["search_offset"] = True
        out.update(changed=True, what="re-search frame_offset (+-5)")
    elif "meas_median" in a:
        p["meas_median"] = max(3, int(p.get("meas_median", 3)) - 2)
        out.update(changed=True, what="meas_median -> %d" % p["meas_median"])
    elif "box_sym_snap" in a:
        if p.get("box_sym_snap") != "continuity":
            p["box_sym_snap"] = "continuity"
            out.update(changed=True, what="box_sym_snap -> continuity")
    elif "gravity" in a:
        p["gravity_lock"] = True
        out.update(changed=True, what="enable gravity lock")

    out["params"] = p
    return out


def calibrate_then_fit(tools, state: dict, quiet: bool = False) -> dict:
    """跑 calibrate_hsv(如需) → fit_box_6dof, 返回该轮的 metrics。

    只跑前两级: 对齐与渲染依赖 g1.pkl, 而 P1.4 的验收点(HSV 放宽 / 尺寸错)
    都在 fit 这一级就能判定, 不必先花 20 分钟渲染。
    `--full` 才会继续往下走。
    """
    p = state["params"]
    if not p.get("hsv"):
        cal = tools["calibrate_hsv"](bag=p["bag"], hsv=None,
                                     out_dir=state["out_dir"])
        if cal.get("hsv"):
            p["hsv"] = cal["hsv"]
            state["log"].append({"stage": "calibrate_hsv", **{
                k: cal.get(k) for k in ("ok", "hsv", "source", "coverage_pct")}})
        else:
            return {"_fatal": "calibrate_hsv failed: %s" % cal.get("error")}

    fit = tools["fit_box_6dof"](
        bag=p["bag"], hsv=p["hsv"], out=state["fit_out"],
        box_size=p.get("box_size"), preview=state.get("fit_preview"),
        min_iou=float(p.get("min_iou", 0.55)),
        max_frames=int(p.get("max_frames", 0)),
        out_dir=state["out_dir"])
    state["log"].append({"stage": "fit_box_6dof", "ok": fit.get("ok"),
                         "rc": fit.get("rc"), "secs": fit.get("secs"),
                         "hsv": p["hsv"], "box_size": p.get("box_size"),
                         "quality": fit.get("quality")})
    state["last_fit"] = fit
    return fit.get("metrics") or {}


_fallback_metrics_doc = None


def run_downstream(tools, state: dict, metrics: dict, out: dict) -> dict:
    """fit 判为 ok 之后, 继续 align -> render -> qa。

    ★ 为什么必须等 fit = ok 才往下走:
      align/render 都吃 fit 的 npz。带着一个 4% 有效率的 npz 去对齐+渲染,
      只会把 30 分钟花在一个注定失败的片子上, 然后在最末端报一个
      无法归因的 QA FAIL —— 那是"编排"而不是"归因"。
    ★ 每一级都**独立诊断**: 下游失败也要给出分类与是否问人的结论,
      而不是笼统地报"渲染失败"。
    """
    p = state["params"]
    d = out["downstream"] = {}

    # ---- align_to_world ----
    al = tools["align_to_world"](
        robot_motion=p["robot"], box_traj=state["fit_out"],
        out=state["world_out"], stool_h=float(p.get("stool_h", 0.45)),
        out_dir=state["out_dir"])
    d["align_to_world"] = {"ok": al.get("ok"), "rc": al.get("rc"),
                           "secs": al.get("secs"), "log": al.get("log"),
                           "quality": al.get("quality")}
    if not al.get("ok"):
        v = tools["diagnose"]({**(al.get("metrics") or {}),
                               "stage": "align"})
        d["align_to_world"]["verdict"] = {"cls": v.get("cls"), "ask": v.get("ask"),
                                          "action": v.get("action"),
                                          "note": v.get("note")}
        out["stopped_at"] = "align 阶段失败: %s" % v.get("cls")
        out["final_verdict"] = v
        return d

    # ---- render_and_qa ----
    # 参数解析次序(后者覆盖前者):
    #   1) RENDER_TUNED: 交付片 v29 用的一整套(run_v29.ps1 逐字), 见下方常量
    #   2) state["params"]["render_extra"]: 若显式给了就整体覆盖
    # ★ 实测教训(2026-10-07): 早前只传"最小四件套"(box_size /
    #   quat_max_step_deg / box_sym_snap / hand_roll_deg), 结果 **复现不出交付片** ——
    #   GRIP CLAMP 的漂移上限从 450mm 掉到 50mm, QA 失败点从 f159-164
    #   变成 f280-281, 最终判定从 qa_known_acceptable 变成 qa_fail_review。
    #   渲染器默认值 ≠ 交付参数, 所以 --full 必须默认带上 v29 的整套。
    #   想跑"裸默认"渲染就传 render_extra=[]。
    extra = list(RENDER_TUNED)
    if p.get("box_size"):
        bsz = str(p["box_size"]).replace(",", " ").split()
        if "--box_size" in extra:
            i = extra.index("--box_size")
            extra[i + 1:i + 4] = [bsz[0], bsz[1], bsz[2]]
    if p.get("render_extra") is not None:
        extra = list(p["render_extra"])
    rr = tools["render_and_qa"](
        robot_motion=p["robot"], box_pose_world=state["world_out"],
        video=state["video_out"], dump_box_pose=state["boxpose_out"],
        extra_args=extra or None, out_dir=state["out_dir"])
    d["render_and_qa"] = {"ok": rr.get("ok"), "rc": rr.get("rc"),
                          "render_secs": rr.get("render_secs"),
                          "qa_secs": rr.get("qa_secs"),
                          "render_log": rr.get("render_log"),
                          "qa_log": rr.get("qa_log"),
                          "carry_rot_max": rr.get("carry_rot_max"),
                          "carry_rot_max_src": rr.get("carry_rot_max_src"),
                          "n_render_args": len(extra),
                          "quality": rr.get("quality")}

    # ---- 用 QA 指标 + world 指标再诊断一次(整链视角) ----
    merged = {}
    merged.update(metrics or {})
    merged.update(rr.get("metrics") or {})
    merged.update(al.get("metrics") or {})
    v = tools["diagnose"](merged)
    d["render_and_qa"]["verdict"] = {"cls": v.get("cls"), "ask": v.get("ask"),
                                     "action": v.get("action"),
                                     "note": v.get("note"),
                                     "evidence": v.get("evidence")}
    out["final_verdict"] = v
    out["final_cls"] = v.get("cls")
    return d


def run_agent(tools, state: dict, full: bool = False) -> dict:
    """主循环: 诊断 -> (按铁律)决定重试或停下。

    `full=True` 时, fit 判为 ok 后继续跑 align -> render -> qa (见 run_downstream)。
    """
    attempt = 0
    history = []
    out = {"history": history, "downstream": None}
    while True:
        metrics = calibrate_then_fit(tools, state)
        if isinstance(metrics, dict) and metrics.get("_fatal"):
            v = {"cls": "tool_failure", "action": None, "ask": True,
                 "evidence": metrics, "note": metrics["_fatal"]}
        else:
            try:
                v = tools["diagnose"](metrics)
            except Exception as e:                             # noqa: BLE001
                # fit 日志被块缓冲截断时(常见于 *> 重定向 + 进程被中断),
                # 指标可能全空。这时**不要**误判成 fit_crashed 去重跑,
                # 而是从 npz 兜底取指标(见 _fallback_metrics)。
                fb = _fallback_metrics(state)
                if fb:
                    v = tools["diagnose"](fb)
                else:
                    v = {"cls": "metrics_unavailable", "action": None,
                         "ask": True, "evidence": {"error": str(e)},
                         "note": "日志解析与 npz 兜底都拿不到指标"}
        rec = {"attempt": attempt, "cls": v.get("cls"),
               "ask": v.get("ask"), "action": v.get("action"),
               "hsv": state["params"].get("hsv"),
               "box_size": state["params"].get("box_size"),
               "evidence": v.get("evidence")}
        history.append(rec)

        # ---- 铁律 1: ask=True 绝不自动重试 ----
        if v.get("ask"):
            rec["stop"] = "STOP-and-ASK (铁律1): 停下问人, 不自动重试"
            break
        if v.get("cls") == "ok":
            rec["stop"] = "通过"
            # fit 通过后才下探 align/render/qa
            if full:
                if not state["params"].get("robot"):
                    rec["stop"] = "通过 (但因缺 --robot, 未跑下游 align/render/qa)"
                    out["final_cls"] = "ok"
                else:
                    run_downstream(tools, state, metrics, out)
                    rec["downstream_ran"] = True
            else:
                out["final_cls"] = "ok"
            break
        if v.get("cls") not in RETRYABLE or not v.get("action"):
            rec["stop"] = "无可用动作 => 升级为问人"
            break

        # ---- 铁律 3: 重试上限 ----
        if attempt >= MAX_RETRIES:
            rec["stop"] = ("升级为问人 (铁律3): 已重试 %d 次" % MAX_RETRIES)
            break

        # ---- 铁律 2: 重试必须改变一件具体的事 ----
        ch = apply_action(v["action"], state)
        rec["change"] = {"what": ch["what"], "changed": ch["changed"]}
        if not ch["changed"]:
            rec["stop"] = ("停止 (铁律2): 该动作没有改变任何参数, "
                           "再跑一次等于重复劳动")
            break
        state["params"] = ch["params"]
        attempt += 1

    out.update({"stopped_at": history[-1].get("stop"),
                "n_attempts": len(history),
                "final_hsv": state["params"].get("hsv"),
                "final_box_size": state["params"].get("box_size")})
    return out


def _fallback_metrics(state: dict) -> dict:
    """日志解析不出指标时, 从 fit npz 兜底(它才是权威产物)。"""
    fit = state.get("last_fit") or {}
    m = dict(fit.get("metrics") or {})
    m.pop("reject_reasons", None)
    return m if m.get("fit_ok_ratio") is not None else {}


# ---------------------------------------------------------------------------
# 离线回放: 用已有故障日志验证归因与重试决策(不启动重活, 秒级)
# ---------------------------------------------------------------------------

def replay(fit_log: str, hsv: str = "4,120,130:14,255,255",
           box_size: str = "0.30,0.30,0.45", verbose: bool = True) -> dict:
    """把一次**已有的** fit 日志喂给诊断+重试决策, 打印它会怎么做。

    这是验证 P1.4 验收标准最快的方式, 也天然满足"必须在真实失败上验证":
    日志就是真实运行产生的。
    """
    m = parse_fit_log(fit_log)
    v = diagnose(m)
    state = {"params": {"bag": "(replay)", "hsv": hsv, "box_size": box_size},
             "out_dir": os.path.join(REPO, "agent", "out"),
             "log": []}
    rec = {"attempt": 0, "cls": v.cls, "ask": v.ask, "action": v.action,
           "hsv": hsv, "box_size": box_size,
           "evidence": {k: v.evidence.get(k) for k in
                        ("fit_ok_ratio", "iou_median", "depth_res_p90_mm",
                         "bbox_w_m", "bbox_inverted", "frac_reject_mask")}}
    if verbose:
        print("  source log : %s" % os.path.relpath(fit_log, REPO))
        print("  metrics    : fit_ok_ratio=%s iou_med=%s depth_p90=%smm" %
              (m.get("fit_ok_ratio"), m.get("iou_median"),
               m.get("depth_res_p90_mm")))
        print("  >= %-18s ask=%-5s" % (v.cls, v.ask))
        print("     why  : %s" % v.note)
    if v.ask:
        rec["stop"] = "STOP-and-ASK (铁律1)"
        rec["would_do"] = "停下问人, 不自动重试"
    elif v.cls == "ok":
        rec["stop"] = "通过"
        rec["would_do"] = "无动作"
    else:
        ch = apply_action(v.action, state)
        rec["change"] = ch["what"]
        rec["changed"] = ch["changed"]
        rec["stop"] = ("自动重试 1 次" if ch["changed"]
                       else "停止(铁律2: 动作未改变任何参数)")
        rec["would_do"] = ch["what"] or "(无改动)"
    if verbose:
        print("     -> do: %s" % rec["would_do"])
    return rec


def verify_acceptance() -> int:
    """P1.4 的两条验收, 用真实故障日志离线验证。"""
    print("=" * 74)
    print("P1.4 验收 A: 给一个「HSV 标错」的输入, Agent 能自动放宽 HSV")
    print("=" * 74)
    a = replay(os.path.join(REPO, "agent", "fixtures", "A6b.log"),
               hsv="4,120,130:14,255,255")
    okA = (a.get("cls") == "mask_too_small"
           and a.get("changed") is True
           and "hsv" in (a.get("change") or ""))
    print("  => %s\n" % ("PASS" if okA else "FAIL"))

    print("=" * 74)
    print("P1.4 验收 B: 给一个「box_size 标错」的输入, Agent 能识别并停下问人")
    print("=" * 74)
    b = replay(os.path.join(REPO, "agent", "fixtures", "BADSIZE.log"),
               box_size="0.20,0.20,0.30")
    okB = (b.get("cls") == "metric_mismatch" and b.get("ask") is True)
    print("  => %s\n" % ("PASS" if okB else "FAIL"))

    print("=" * 74)
    print("对照: 健康的 165945 基线")
    print("=" * 74)
    c = replay(os.path.join(REPO, "agent", "fixtures",
                            "165945_fit.log"))
    print()
    print("=" * 74)
    print("结果: A=%s B=%s  => %s" %
          ("PASS" if okA else "FAIL", "PASS" if okB else "FAIL",
           "两条验收都通过" if (okA and okB) else "有验收未通过"))
    print("=" * 74)
    return 0 if (okA and okB) else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag")
    ap.add_argument("--robot", help="data/<prefix>_g1.pkl (full 模式需要)")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--hsv", default=None)
    ap.add_argument("--box-size", default="0.30,0.30,0.45")
    ap.add_argument("--stool-h", type=float, default=0.45)
    ap.add_argument("--fit-out", default=None)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="冒烟用: 只跑前 N 帧(0=全部)")
    ap.add_argument("--full", action="store_true",
                    help="继续跑 align + render + qa(需 --robot)")
    ap.add_argument("--replay", help="离线回放一个 fit 日志")
    ap.add_argument("--verify", action="store_true",
                    help="跑 P1.4 的两条验收(离线, 秒级)")
    args = ap.parse_args()

    if args.verify:
        sys.exit(verify_acceptance())
    if args.replay:
        r = replay(args.replay, hsv=args.hsv or "4,120,130:14,255,255",
                   box_size=args.box_size)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    if not args.bag:
        ap.error("need --bag (or --replay/--verify)")
    if args.full and not args.robot:
        ap.error("--full needs --robot data/<prefix>_g1.pkl")

    out_dir = os.path.join(REPO, "agent", "out")
    os.makedirs(out_dir, exist_ok=True)
    pref = args.prefix or "(unknown)"
    state = {
        "params": {"bag": args.bag, "hsv": args.hsv, "box_size": args.box_size,
                   "robot": args.robot, "stool_h": args.stool_h,
                   "min_iou": 0.55, "max_frames": args.max_frames,
                   "meas_median": 3, "box_sym_snap": "continuity",
                   "quat_max_step_deg": 20,   # 交付片 v29 用的值(默认 3.5 会裁掉真转动)
                   "hand_roll_deg": 0},
        "out_dir": out_dir,
        "fit_out": args.fit_out or os.path.join(
            out_dir, "%s_6dof.npz" % pref, ),
        "fit_preview": os.path.join(out_dir, "%s_6dof.mp4" % pref),
        # ---- 下游产物 (--full) ----
        "world_out": os.path.join(out_dir, "%s_world_6dof.npz" % pref),
        "boxpose_out": os.path.join(out_dir, "%s_boxpose_used.npz" % pref),
        "video_out": os.path.join(out_dir, "%s_measured_box_6dof.mp4" % pref),
        "log": [],
    }
    import tools as T
    t0 = time.time()
    res = run_agent(T.TOOLS, state, full=args.full)
    res["wall_secs"] = round(time.time() - t0, 1)
    res["state_log"] = state["log"]
    res["fit_out"] = os.path.relpath(state["fit_out"], REPO)
    if res.get("downstream"):
        res["world_out"] = os.path.relpath(state["world_out"], REPO)
        res["video_out"] = os.path.relpath(state["video_out"], REPO)
        res["boxpose_out"] = os.path.relpath(state["boxpose_out"], REPO)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
