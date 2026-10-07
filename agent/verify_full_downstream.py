#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the --full downstream wiring (align -> render -> qa) WITHOUT re-running
the 10-minute fit.

Why: run_agent --full spends ~10 min in fit before it ever reaches the code I
just wrote.  This driver calls run_downstream() directly on the EXISTING
delivered fit npz, so a wiring bug surfaces in minutes instead of half an hour.

It is a real run, not a mock: it executes the actual scripts through the actual
tools layer and produces real artifacts under .workbuddy/agent/out/.

Usage (repo root, gmr python):
    python agent/verify_full_downstream.py
"""
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

import run_agent as RA            # noqa: E402
import tools as T                 # noqa: E402

PREFIX = "FULLTEST"
OUT = os.path.join(REPO, "agent", "out")


def main():
    os.makedirs(OUT, exist_ok=True)

    # 复用**随包样例**的 fit 产物(418 帧, 100% 有效率), 不重跑拟合。
    # ★ 源仓库里这行原本读 data/20260922_165945_6dof.npz —— 那条素材
    #   没有随包分发（体积原因），克隆者一跑就会 sys.exit。
    #   改成 example/ 里随包的那条（20260922_162859），包内自洽。
    fit_src = os.path.join(REPO, "example", "20260922_162859", "data",
                           "20260922_162859_6dof.npz")
    fit_dst = os.path.join(OUT, "%s_6dof.npz" % PREFIX)
    if not os.path.isfile(fit_src):
        sys.exit("missing %s  —— 该文件应随包附带（example/20260922_162859/data/）"
                 % fit_src)
    import shutil
    shutil.copy2(fit_src, fit_dst)
    print("reusing example fit: %s" % os.path.relpath(fit_src, REPO))

    # ★ 用 run_v29.ps1 的**逐字参数**复现交付片。
    #   只给最小默认集会渲染出一个未调参版本(QA 失败点变成 f280),
    #   那不是"接线错了", 而是"没带交付参数"。
    #   该参数集现已抽到 run_agent.RENDER_TUNED, 由 --full 默认使用;
    #   这里显式引用同一个常量, 保证两条路不会漂移。
    RENDER_V29 = list(RA.RENDER_TUNED)

    state = {
        "params": {
            "bag": "(reused)", "hsv": "4,75,79:24,255,255",
            "box_size": "0.30,0.30,0.45",
            "robot": "example/20260922_162859/data/20260922_162859_g1.pkl",
            "stool_h": 0.45, "quat_max_step_deg": 20,
            "box_sym_snap": "continuity", "hand_roll_deg": 0,
            "render_extra": RENDER_V29,
        },
        "out_dir": OUT,
        "fit_out": fit_dst,
        "fit_preview": None,
        "world_out": os.path.join(OUT, "%s_world_6dof.npz" % PREFIX),
        "boxpose_out": os.path.join(OUT, "%s_boxpose_used.npz" % PREFIX),
        "video_out": os.path.join(OUT, "%s_measured_box_6dof.mp4" % PREFIX),
        "log": [],
    }

    # 已知的 fit 指标(来自交付 npz), 下游诊断要它做整链视角
    metrics = T._npz_fit_metrics(fit_dst)
    print("fit metrics: fit_ok=%s iou_med=%s depth_med=%smm"
          % (metrics.get("fit_ok"), metrics.get("iou_median"),
             metrics.get("depth_res_med_mm")))

    out = {"history": [], "downstream": None}
    t0 = time.time()
    RA.run_downstream(T.TOOLS, state, metrics, out)
    out["wall_secs"] = round(time.time() - t0, 1)

    print("\n" + "=" * 74)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print("=" * 74)

    d = out.get("downstream") or {}
    v = (d.get("render_and_qa") or {}).get("verdict") or {}
    ok = bool(d.get("align_to_world", {}).get("ok")
              and d.get("render_and_qa", {}).get("ok") is not None)
    print("align ok      :", d.get("align_to_world", {}).get("ok"))
    print("render+qa ran :", d.get("render_and_qa", {}).get("ok") is not None)
    print("final verdict :", v.get("cls"), "ask=", v.get("ask"))
    print("QA result     :", (d.get("render_and_qa", {}).get("quality") or {}))
    print("=> downstream wiring %s" % ("OK" if ok else "BROKEN"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
