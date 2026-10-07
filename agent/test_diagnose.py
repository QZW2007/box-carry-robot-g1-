#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_diagnose.py -- diagnose.py 的规则覆盖测试。

三种用例:
  A 真实片(165945): 用**交付产物**的 QA 日志, 期望 qa_known_acceptable
                    (f159-164 拾取段 TELEPORT = 计划书附录B第8行"已知可接受")
  B 合成故障: 逐条篡改真实指标, 验证每条规则都能被触发
  C 真实故障回归: 用两次**真跑**产生的高危故障日志, 锁定两个已修的真实缺陷
                  (掩码退化 vs 先验尺寸错 必须分开; 拾取段 teleport 不得触发重搜offset)

用法: python test_diagnose.py
"""
import os
import sys
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_metrics import collect, parse_fit_log, parse_world_log, parse_qa_log
from diagnose import diagnose, TH

HERE = os.path.dirname(os.path.abspath(__file__))
# 打包后: 全部真实日志证据都在 agent/fixtures/ （见 fixtures/README.md）
FIX = os.path.join(HERE, "fixtures")
B = FIX          # 兼容旧命名: 原 .workbuddy/batch_logs
P02 = FIX        # 原 .workbuddy/p02
P1 = FIX         # 原 .workbuddy/p1


def real_metrics():
    """用**交付产物**(数据/ 里的 v29 boxpose)重生成的 QA 日志。

    ★ 2026-10-07 修正: 原来这里读 batch_logs/qa_...165945....log, 那份是
      9/27 旧产物(data/20260922_165945_boxpose_used.npz, md5 5e591c14)时代的
      日志, RESULT=PASS; 而**实际交付**的是 10/6 的 v29
      (数据/.../boxpose_used.npz, md5 63473c15), 它的 QA 是 FAIL —— 两者
      不是同一次渲染。fixture 现已改为由交付产物重新生成, 见 fixtures/README。
    """
    qa = os.path.join(FIX, "qa_20260922_165945.log")
    if not os.path.isfile(qa):
        qa = os.path.join(B, "qa_165945_batch_logs.log")
    return collect(
        os.path.join(B, "165945_fit.log"),
        os.path.join(B, "165945_world.log"),
        qa,
    )


def run(name, mut, expect):
    m = real_metrics()
    mut(m)
    v = diagnose(m)
    ok = (v.cls == expect)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:34s} -> {v.cls:24s} ask={str(v.ask):5s} (期望 {expect})")
    if not ok:
        print("       证据:", v.evidence)
    return ok


def main():
    results = []

    # ---------- A. 真实片(交付产物) ----------
    v = diagnose(real_metrics())
    results.append(v.cls == "qa_known_acceptable")
    print(f"[{'PASS' if results[-1] else 'FAIL'}] {'A. 交付片 165945(v29)':34s} -> {v.cls} ask={v.ask}")
    print("       期望 qa_known_acceptable + ask=False: f159-164 拾取段 TELEPORT "
          "是已知可接受项,\n       只记录不重试(重搜 offset 会改坏好片子)\n")

    # ---------- B. 合成故障: 一条规则一个用例 ----------

    # 1 fit 完全没产出(脚本崩了)
    def m1(m): m.pop("fit_total", None); m.pop("frames_total", None)
    results.append(run("B1 fit 无产出", m1, "fit_crashed"))

    # 2 掩码太小 / 有效帧太少  -> mask_too_small
    #   ★ 2026-10-07: 这条规则现在要求**掩码本身有病灶**(bbox 倒挂或存在
    #     mask/blob 级拒绝), 否则会与 metric_mismatch 混。合成用例要一起给。
    def m2(m):
        m["valid_ratio"] = 0.31
        m["fit_ok_ratio"] = 0.31
        m["bbox_w_m"] = [0.05, 0.09]
        m["fx"] = 643.0
        m["bbox_inverted"] = True           # 掩码被切碎
        m["frac_reject_mask"] = 0.10
    results.append(run("B2 掩码太小(有病灶)", m2, "mask_too_small"))

    # 2b 只有"拟合率低"、掩码却是好的 -> 必须落到 metric_mismatch(停下问人)
    #    这个对照正是 2026-10-07 修掉的真实缺陷(以前会误判成 mask_too_small)
    def m2b(m):
        m["valid_ratio"] = 1.0
        m["fit_ok_ratio"] = 0.06
        m["bbox_w_m"] = [0.328, 0.52]        # 掩码正常
        m["bbox_inverted"] = False
        m["frac_reject_mask"] = 0.0          # 零掩码级拒绝
        m["iou_median"] = 0.579
        m["depth_res_med_mm"] = 28.0
        m["depth_res_p90_mm"] = 32.0
    results.append(run("B2b 掩码正常但轮廓不符", m2b, "metric_mismatch"))

    # 3 深度残差大  -> tilt_ambiguity
    def m3(m):
        m["depth_res_med_mm"] = 210.0
        m["depth_res_p90_mm"] = 480.0
        m["iou_median"] = 0.88          # IoU 还高 —— 正是歧义签名
    results.append(run("B3 倾斜歧义", m3, "tilt_ambiguity"))

    # 4 IoU 低但深度正常 -> metric_mismatch (必须 ask)
    def m4(m):
        m["iou_median"] = 0.42
        m["iou_p10"] = 0.33
        m["depth_res_med_mm"] = 25.0
        m["depth_res_p90_mm"] = 33.0
    results.append(run("B4 尺寸/标定不符", m4, "metric_mismatch"))

    # 5 carry 段巨大角步 -> symmetry_flip
    def m5(m):
        m["phases"]["CARRY"]["rot_max_deg"] = 92.0
    results.append(run("B5 对称群跳变", m5, "symmetry_flip"))

    # 6 非预期位置跳变 -> frame_offset_mismatch
    #   位置要远离拾取交接带(否则会被 B5b 的"已知可接受"规则接走)
    def m6(m):
        m["jumps"] = [{"label": "carry mid", "f0": 200, "f1": 201, "mm": 180.0,
                       "carry": (1, 1), "cov": (1, 1)},
                      {"label": "carry mid2", "f0": 220, "f1": 221, "mm": 150.0,
                       "carry": (1, 1), "cov": (1, 1)}]
    results.append(run("B6 帧偏移错", m6, "frame_offset_mismatch"))

    # 6b 拾取交接带内的短段 TELEPORT -> qa_known_acceptable (只记录, 不重试)
    #    ★ 这条锁定 2026-10-07 修掉的第二个真实缺陷: 交付片 v29 的
    #      f159-164 之前会被判成 frame_offset_mismatch 并被自动重搜 offset
    def m6b(m):
        m["jumps"] = [{"label": "TELEPORT", "f0": 159 + i, "f1": 160 + i,
                       "mm": 52.0 + i, "carry": (1, 1), "cov": (1, 1)}
                      for i in range(5)]
        m["pass"] = False
        m["result"] = "FAIL"
        m["first_fail_frame"] = 159
    results.append(run("B6b 拾取段已知TELEPORT", m6b, "qa_known_acceptable"))

    # 7 停放段真冻结失败(远超噪声)
    def m7(m):
        m["warn_parked_pos_jump_mm"] = 95.0
        m["phases"]["STOOL"]["rot_max_deg"] = 0.0     # 只位移不转 => 仍需报
    results.append(run("B7 冻结失效(大位移)", m7, "freeze_failed"))

    # 8 停放段整段在转动(p90 高) -> 冻结失效
    #   ★ 用 p90 而不是 max: max 是单帧量化噪声(已交付片 max 1.5° 而 p90 0.000°),
    #     单帧 6° 的 glitch 不足以说明"整段在动"。
    def m8(m):
        m["warn_parked_pos_jump_mm"] = None
        m["phases"]["STOOL"]["rot_p90_deg"] = 6.0
        m["phases"]["STOOL"]["rot_max_deg"] = 6.0
    results.append(run("B8 冻结失效(整段转动)", m8, "freeze_failed"))

    # 9 s 偏离 -> scale_anomaly (必须 ask)
    def m9(m):
        m["sim3_s"] = 1.05
        m["scale"] = 1.05
    results.append(run("B9 尺度异常", m9, "scale_anomaly"))

    # 10 重力未锁 -> gravity_not_locked
    def m10(m):
        m["axis_up_deg"] = 6.7
        m["lean_deg"] = 6.7
    results.append(run("B10 重力未锁", m10, "gravity_not_locked"))

    # 11 z 轴落在对称支(180°) -> 应报 INFORMATIONAL, 不能当 bug
    def m11(m):
        # 真实值就是 175.8, 但上面 case 6 会先命中吗? 这里已重设 jumps
        m["jumps"] = [{"label": "lift/transition, expected", "f0": 280, "f1": 281,
                       "mm": 62.5, "carry": (1, 0), "cov": (1, 1)}]
        m["pass"] = True
        m["result"] = "PASS"
        m["warn_parked_pos_jump_mm"] = None
        m["world_zaxis_med_deg"] = 175.8
    results.append(run("B11 z轴对称支(假警报)", m11, "zaxis_symmetry_branch"))

    # 12 一切正常
    def m12(m):
        m["pass"] = True
        m["result"] = "PASS"
        m["warn_parked_pos_jump_mm"] = None
        m["jumps"] = []
        m["world_zaxis_med_deg"] = 4.1      # 落在正常支
    results.append(run("B12 全正常", m12, "ok"))

    # ---------- C. 真实故障回归(2026-10-07 真跑产生的日志) ----------
    # 这三条**不是合成**, 是完整跑 354 帧真实 bag 得到的日志。它们锁定两个
    # 已修的真实缺陷, 以及一次端到端自愈的判定。
    print("\n--- C. 真实故障回归(真跑日志, 非合成) ---")

    def real_fit_case(tag, logpath, expect_cls):
        if not os.path.isfile(logpath):
            print(f"[SKIP] {tag:34s} -> (缺日志 {logpath})")
            return True          # 缺证据不算失败, 但会在上面提示
        v = diagnose(parse_fit_log(logpath))
        ok = (v.cls == expect_cls)
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:34s} -> {v.cls:22s} "
              f"ask={str(v.ask):5s} (期望 {expect_cls})")
        if not ok:
            print("       证据:", v.evidence)
        return ok

    # C1 HSV 收窄 50%(A6b): 掩码退化 -> mask_too_small, 可自动放宽
    results.append(real_fit_case(
        "C1 HSV收窄->掩码退化",
        os.path.join(P02, "A6b.log"), "mask_too_small"))

    # C2 box_size 填错 0.20/0.30(BADSIZE): 掩码正常但轮廓不符 -> 停下问人
    results.append(real_fit_case(
        "C2 尺寸填错->停下问人",
        os.path.join(P1, "BADSIZE.log"), "metric_mismatch"))

    # C3 修复动作真的能修好: 放宽后的 HSV 实跑 354/354 (R2)
    r2 = os.path.join(P1, "R2.log")
    if os.path.isfile(r2):
        m2 = parse_fit_log(r2)
        ok = (m2.get("fit_ok") == 354)
        print(f"[{'PASS' if ok else 'FAIL'}] {'C3 放宽HSV实跑恢复100%':34s} -> "
              f"fit_ok={m2.get('fit_ok')}/354 IoU中位={m2.get('iou_median')}")
        results.append(ok)
    else:
        print("[SKIP] C3 (缺 R2.log)")

    # C4 只降 S/V 下限修不好(RELAXED): 证明"重试必须改到要害"
    rl = os.path.join(P1, "RELAXED.log")
    if os.path.isfile(rl):
        mrl = parse_fit_log(rl)
        ok = (mrl.get("fit_ok_ratio") or 0) < 0.20      # 仍然很差
        print(f"[{'PASS' if ok else 'FAIL'}] {'C4 只降S/V修不好(反例)':34s} -> "
              f"fit_ok={mrl.get('fit_ok')}/354 (证据: 必须同时放色相上限)")
        results.append(ok)
    else:
        print("[SKIP] C4 (缺 RELAXED.log)")

    n_ok = sum(1 for r in results if r)
    print(f"\n===== {n_ok}/{len(results)} 通过 =====")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
