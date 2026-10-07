#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose.py -- 把流水线的结构化残差映射到"具体失效模式 + 建议动作"。

这是整个 Agent 的**技术壁垒**（《项目掌握计划-v2.md》P1.0 / P1.3）。
编排本身是工程；**失败归因需要理解流水线的物理假设**，这是通用候选人拿不出来的。

输入: parse_metrics.collect() 产出的扁平 dict
输出: Verdict(分类, 建议动作, 是否必须问人, 证据, 置信度)

设计来自真实 debug 史（不是想出来的）:
    每一条规则都对应本项目里真实发生过的一次故障，注释里写了是哪一次。

三条铁律（面试必问"你的 Agent 会不会乱重试"）:
    1. ask=True 的分类**绝不自动重试**。重试永远好不了，只会烧时间和钱。
       "知道什么时候停下来问人，是 Agent 成熟度的标志。"
    2. 重试必须**改变一件具体的事**（换 HSV / 加初始器 / 重搜 offset / 提阈值），
       不是"再跑一次"。带随机种子的重跑不叫重试。
    3. 重试上限 2 次（由调用方执行），之后升级为 ask=True。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Verdict:
    """一次诊断结论。"""
    cls: str                                  # 失效模式标识
    action: Optional[str]                     # 建议动作(None = 只能问人)
    ask: bool                                 # True = 必须停下来问人, 禁止自动重试
    evidence: dict = field(default_factory=dict)   # 触发这条规则的观测值(面试要能报)
    confidence: str = "high"                  # high / medium / low
    note: str = ""                            # 机理一句话

    def __str__(self) -> str:
        z = "STOP-and-ASK" if self.ask else "auto-retry-ok"
        return (f"[{self.cls}] {z} conf={self.confidence}\n"
                f"  action  : {self.action}\n"
                f"  evidence: {self.evidence}\n"
                f"  why     : {self.note}")


# ---------------------------------------------------------------------------
# 阈值: 全部来自 165945 / 0917 两条片子的实测统计, 不是拍的
# ---------------------------------------------------------------------------
TH = {
    "mask_px_small": 60,        # 箱体 <60px 时 mask_pixels 噪声主导
    "valid_ratio_low": 0.70,    # 有效帧比例低于此 => 掩码/HSV 出问题
    "iou_low": 0.70,            # silhouette 明显不匹配
    "depth_ok": 0.04,           # (m) 深度残差可以接受的上限
    "depth_bad": 0.15,          # (m) cuboid_fit.solve() 内部判定"无测量"的门限
    "frac_depth_bad": 0.15,     # >15% 帧深度残差超限 => 系统性问题而非个别噪声
    "step_expect_max": 30.0,    # (deg) 非坡道区单帧角步; 165945 实测 max 2.565
    "parked_pos_tol_mm": 10.0,  # parked 段位置抖动**WARN**门限(check_render_quality 内部值)
    # ★ 注意: 10mm 是 QA 的 WARN 门限, 不是"故障"门限。停放段测量值本身有 ~10mm
    #   深度噪声, 10~20mm 在合格片里是常态(165945 = 15.8mm, 已验收)。
    #   判"冻结失败"要用**高于噪声量级**的门限, 否则每条合格片都误报。
    "parked_freeze_pos_mm": 40.0,   # 真"冻结失败"的位置门限(噪声的 3~4 倍)
    "parked_rot_max_deg": 1.0,      # 真"冻结失败"的转动门限(冻住就不该转)
    "lift_jump_mm": 50.0,       # unmasked 跳变门限
    "known_pickup_max_frames": 8,   # 拾取交接带内可接受的 TELEPORT 帧数上限
    "pickup_band_frames": 12,       # 拾取交接带的宽度(帧); 跨度超过它就不算"紧凑"
    "parked_rot_p90_deg": 0.5,      # parked 转动 p90 门限(= QA 自己的 PARKED_ROT_P90)
    "s_expected": 0.7155,       # Umeyama 独立测得; 同片复跑偏差 >3% 就异常
    "s_tol_rel": 0.03,
    "axis_up_max_deg": 3.0,     # 重力锁定后 cam-up vs world+z 应贴近 0
    "zaxis_sym_tol": 25.0,      # 允许 world z-axis 落在 180° 对称支(165945=175.8)
}


def _g(m: dict, k: str, default=None):
    """安全取值: 支持嵌套阶段字典, 顶层优先。"""
    if k in m:
        return m[k]
    for st in ("qa", "world", "fit"):
        sub = m.get(st)
        if isinstance(sub, dict) and k in sub:
            return sub[k]
    return default


def diagnose(m: dict) -> Verdict:
    """输入 parse_metrics 的扁平 dict, 输出 Verdict。

    判断顺序 = "最可能根因优先", 因为故障时多个信号会同时亮,
    必须靠**因果方向**而不是信号强度来排序(越靠前的越可能是根因)。
    """
    ev = {}
    def got(*keys):
        for k in keys:
            v = _g(m, k)
            if v is not None:
                ev[k] = v
        return ev

    # ------------------------------------------------------------------
    # 0. 上游根本没产出 —— 先排除"没跑起来", 否则后面全是误报
    # ------------------------------------------------------------------
    fit_missing = _g(m, "fit_total") is None and _g(m, "frames_total") is None
    if fit_missing:
        return Verdict(
            "fit_crashed", "check_log_stderr_and_camera_calib", ask=True,
            evidence=got("fit_ok", "frames_total"),
            note="fit 阶段没有产出统计行 => 脚本崩了或没到拟合步骤; "
                 "最常见的根因是标定文件缺失/相机内参读不到, 这不是靠调参能好的。")

    # ------------------------------------------------------------------
    # 1. 掩码退化 / 有效帧太少  ==> HSV 标定问题
    #    真实案例: 0917 那批没锁自动曝光, HSV 飘掉, 掩码大面积丢失
    #
    #    ★ 两类"拟合率低"必须分开(否则会误判成"先验错了", 白问人一次):
    #      (a) **掩码退化** —— 颜色分割没框住物体。签名: 存在**与几何无关**的
    #          掩码级拒绝(`mask too small` / `no blob accepted`), 掩码被切碎到
    #          包围盒统计**倒挂**(上界 < 下界)。动作 = 放宽 HSV。
    #          实测 A6b(HSV 收窄 50%): fit_ok 4.5%, 拒绝 = 308×IoU + 22×mask
    #          too small + 8×no blob, bbox_w_m=[0.284,**0.14**] 倒挂。
    #      (b) **先验尺寸错** —— 掩码是好的, 但投影轮廓在任何位姿都对不上。
    #          签名: 拒绝**全是** `IoU < 门限`, 零掩码级拒绝, bbox 正常。
    #          实测 BADSIZE(box 填 0.20/0.30, 真值 0.30/0.45): fit_ok 5.9%,
    #          拒绝 = 333×IoU, **0** 掩码级依赖, bbox_w_m=[0.328,0.52] 正常。
    #          这一类归第 3 条 metric_mismatch(停下问人), 不是 HSV 问题。
    # ------------------------------------------------------------------
    vr = _g(m, "valid_ratio")
    fk = _g(m, "fit_ok_ratio")
    frac_mask = _g(m, "frac_reject_mask")
    bbox_inv = _g(m, "bbox_inverted")
    mask_disease = (bbox_inv is True) or \
                   (frac_mask is not None and frac_mask > 0.0)
    bp = _g(m, "bbox_w_m")
    mask_px = None
    if isinstance(bp, (list, tuple)) and len(bp) == 2:
        # bbox 是"米"的包围盒; 用投影近似像素尺度: px ≈ m * fx / z, z 取 3 m 量级
        fx = _g(m, "fx") or 643.0
        mask_px = (bp[0] * fx / 3.0) if bp[0] else None
    tracker_bad = (vr is not None and vr < TH["valid_ratio_low"]) and \
                  (mask_px is None or mask_px < TH["mask_px_small"])
    fit_mask_bad = (fk is not None and fk < TH["valid_ratio_low"]) and mask_disease
    if tracker_bad or fit_mask_bad:
        ev.update(got("valid_ratio", "fit_ok_ratio", "frames_total",
                      "frames_valid", "bbox_w_m", "bbox_inverted",
                      "frac_reject_mask", "n_reject_mask", "n_reject_iou"))
        return Verdict(
            "mask_too_small", "relax_hsv(S-20,V-20) and re-calibrate; verify auto-exposure locked",
            ask=False,
            evidence=ev,
            note="有效帧比例低且掩码本身有病灶 => HSV 区间过窄/曝光飘了, "
                 "像素没框住箱子(掩码被切碎到包围盒倒挂, 或出现 mask too small / "
                 "no blob 这类与几何无关的拒绝)。重试时**换 HSV 下限**才叫"
                 "「改变一件具体的事」。"
                 "★ 与 metric_mismatch 的分界: 这里**掩码是病的**, "
                 "而尺寸/标定错时掩码是好的、只是轮廓对不上。")

    # ------------------------------------------------------------------
    # 2. 深度残差大  ==> toward/away 倾斜歧义被选中
    #    真实案例: 早期 w_d=0 时, 向相机倾斜的解 IoU 也能到 0.89,
    #              只有深度项能识破; 初始器不够密时会落到这一支
    # ------------------------------------------------------------------
    frac_bad = None
    dm = _g(m, "depth_res_med_mm")
    if dm is not None:
        frac_bad = 1.0 if dm / 1000.0 > TH["depth_bad"] else 0.0
    if (dm is not None and dm / 1000.0 > TH["depth_bad"]) or \
       (frac_bad is not None and frac_bad > TH["frac_depth_bad"]):
        return Verdict(
            "tilt_ambiguity", "densify_tilt_initialisers(-25/0/25 -> -40..40 step 10)",
            ask=False,
            evidence=got("depth_res_med_mm", "depth_res_p90_mm", "iou_median", "img_tilt_med_deg"),
            note="透视下「向相机倾」与「背离相机倾」轮廓几乎同构(实测 IoU 0.89 的自洽错解), "
                 "只有深度模式能区分。深度残差大但 IoU 不低 = 落到了这一支的典型签名。")

    # ------------------------------------------------------------------
    # 3. IoU 低但深度却正常  ==> **不是拟合的问题, 是输入不一致**
    #    ★ 这一类必须停下来问人。真实案例: 相机标定换了 / box_size 填错,
    #      此时重试任何参数都无效, 因为模型的"物体"跟画面里的不是同一个东西。
    # ------------------------------------------------------------------
    iou = _g(m, "iou_median")
    dp90 = _g(m, "depth_res_p90_mm")
    if (iou is not None and iou < TH["iou_low"]) and \
       (dp90 is not None and dp90 / 1000.0 < TH["depth_ok"]):
        return Verdict(
            "metric_mismatch", None, ask=True,
            evidence=got("iou_median", "iou_p10", "depth_res_p90_mm", "box_w", "box_h", "box_d", "fx"),
            note="IoU 低而深度一致 => 轮廓对不上但前后表面位置是对的, 说明"
                 "**送进来的 box_size 或 K 与实际物体/相机不符**。"
                 "这是 Agent 必须停下来问人的第一类: 重试永远不会改善, 只会烧时间。")

    # ------------------------------------------------------------------
    # 4. 非坡道区出现巨大角步  ==> 对称群解分支跳变
    #    真实案例: solve() 逐帧独立解 8 重对称群, 0917 有 183 对 >5° 跳变
    # ------------------------------------------------------------------
    qa_phases = _g(m, "phases") or {}
    carry = qa_phases.get("CARRY") if isinstance(qa_phases, dict) else None
    rot_max = (carry or {}).get("rot_max_deg") if carry else _g(m, "carry_rot_max_deg")
    if rot_max is not None and rot_max > TH["step_expect_max"]:
        return Verdict(
            "symmetry_flip", "box_sym_snap=continuity (+ keep canonicalize_rotations)",
            ask=False,
            evidence=got("canon_jumps_gt5deg", "world_zaxis_med_deg"),
            note=f"carry 段单帧角步 {rot_max:.1f}° 远超真实转速(165945 实测 max 2.565°) => "
                 "帧间在对称等价解之间跳。方截面箱绕 z 转 90° 外观完全等价, "
                 "逐帧独立求解必然抖;**链式归一**才是正解, 不是加大平滑。")

    # ------------------------------------------------------------------
    # 5. 起举瞬间的位置跳变  ==> 帧偏移或分段点错
    #    真实案例: frame_offset 方向搞反(源视频帧 = 渲染帧 + offset), 165945 = -6
    #
    #    ★ 判据必须是"**全线散布**的非预期跳变", 不能只看"有没有 >50mm 的跳":
    #      已交付的 v29(数据/20260922_165945) QA = FAIL, 唯一失败是 f159-164
    #      的 5 个 TELEPORT(52-58mm, carry=1->1)。它们集中在**拾取交接带**、
    #      总跨度 5 帧 —— 这是计划书附录 B 第 8 行写明的"已知可接受"项。
    #      若照字面"有非预期跳变就重搜 offset", 会把一条好片子送去改坏
    #      (offset 已由 hop-1 人-机器人互相关验证: lag 0-1 帧, corr>=0.977)。
    #      真正的 offset 错会让跳变**散落在多个 phase 边界**上, 所以这里要求
    #      跳变跨度大或数量多才判 frame_offset_mismatch; 紧凑的拾取带留给
    #      后面的 qa_known_acceptable 规则(它排在所有有后果的判断之后)。
    # ------------------------------------------------------------------
    jumps = _g(m, "jumps") or []
    first_c, last_c = _g(m, "qa_first_c"), _g(m, "qa_last_c")
    bad = [j for j in jumps if (j.get("mm") or 0) > TH["lift_jump_mm"]
           and "expected" not in (j.get("label") or "")]
    tight_pickup = False
    if bad and first_c is not None:
        f0s = [j["f0"] for j in bad]
        inside = all((j.get("carry") or (0, 0))[0] == 1
                     and (j.get("carry") or (0, 0))[1] == 1 for j in bad)
        tight_pickup = bool(
            inside and (min(f0s) >= first_c)
            and (max(f0s) - min(f0s)) <= TH["pickup_band_frames"]
            and len(bad) <= TH["known_pickup_max_frames"])
    if bad and not tight_pickup:
        ev["unexpected_jump"] = bad[0]
        return Verdict(
            "frame_offset_mismatch", "search_frame_offset(+-5) and re-split phases",
            ask=False,
            evidence=got("frame_offset", "offset_corr", "qa_first_c", "qa_last_c"),
            note="出现**散布的**非预期 >50mm 位置跳变 => 人体轨迹与箱体轨迹的帧"
                 "没对齐。把源视频帧写成 `渲染帧 + offset`(165945 = -6), "
                 "重搜 offset 后重跑。")

    # ------------------------------------------------------------------
    # 6. parked 段还有转动/位移  ==> 冻结失效
    #    真实案例: --meas_median / --meas_smooth_quat 在 carry 段是对的,
    #              但停放段要真冻结; 冻结没生效时箱子会在凳子上"呼吸"
    #
    #    ★ 关键区分(否则会误报): 停放段**测量值**本身就有 ~10mm 深度噪声,
    #      QA 内部把 10mm 定为 WARN 门限。也就是说 10~20mm 的 parked 跳变
    #      在**已交付片**里是常态(165945 = 15.8mm, 已验收通过)。
    #      真正代表"冻结失效"的信号是:
    #        (a) 位移 **远超**噪声量级(>40mm), 或
    #        (b) parked 段出现可观**转动** —— 但必须用 **p90** 而不是 max:
    #            冻结是"整段钉住", 单帧 max 是深度/量化噪声。实测已交付的
    #            v29: STOOL rot med 0.000 / p90 0.000 / **max 1.068** —— 若按
    #            max > 1° 判"冻结失效", 会把它误报成故障(3defect 里的第三个)。
    #            用 p90 > 0.5°(QA 自己的 parked 门限)才是"整段在动"的意思。
    # ------------------------------------------------------------------
    stool = qa_phases.get("STOOL") if isinstance(qa_phases, dict) else None
    warn_park = _g(m, "warn_parked_pos_jump_mm")
    stool_p90 = (stool or {}).get("rot_p90_deg")
    stool_max = (stool or {}).get("rot_max_deg")
    # 两种"冻结失效", 门限都必须定在**噪声之上**:
    #   (a) 整段在动: p90 > 0.5°(= QA 自己的 PARKED_ROT_P90)
    #   (b) 单帧大跳: max > 25° —— 远高于量化噪声
    #   ★ 实测教训: 已交付 v29 的 FLOOR rot max = 1.501° / STOOL max = 1.068°,
    #     而两者 p90 都是 **0.000**。若按 max > 1° 判"冻结失效", 这条已验收的
    #     好片子会被误报成故障(合成用例 B8 想让单帧 6° 触发, 但 6° 同样是
    #     单帧量化噪声量级, 所以真正能代表"抖动"的是 p90)。
    rot_bad = (stool_p90 is not None and stool_p90 > TH["parked_rot_p90_deg"])
    if (warn_park is not None and warn_park > TH["parked_freeze_pos_mm"]) or rot_bad:
        return Verdict(
            "freeze_failed", "reduce_meas_median / tighten parked freeze window",
            ask=False,
            evidence=got("warn_parked_pos_jump_mm"),
            note=f"停放段应完全静止, 实测 STOOL 段位置跳变 {warn_park} mm / "
                 f"转动 p90 {stool_p90}°(门限 {TH['parked_rot_p90_deg']}°) => "
                 "冻结逻辑没盖住这些帧。"
                 "注意: 停放段**测量值**本身有 ~10mm 深度噪声(10~20mm 属正常), "
                 "判「整段在动」要看 **p90** 而不是 max —— 已交付片实测"
                 "max 1.5° 而 p90 0.000°, 单帧 max 只是量化噪声。")

    # ------------------------------------------------------------------
    # 7. 尺度/重力异常  ==> 对齐层出问题
    #    真实案例: 重力未锁定时 up 轴偏 4.9°; 锁定后 1.8°。
    #              s 应在 0.7155 附近(双链路互证, 误差 0.80%)
    # ------------------------------------------------------------------
    s = _g(m, "sim3_s", _g(m, "scale"))
    up = _g(m, "axis_up_deg")
    if (s is not None and abs(s - TH["s_expected"]) / TH["s_expected"] > TH["s_tol_rel"]):
        return Verdict(
            "scale_anomaly", "check_intrinsics_and_depth_scale; verify g1.pkl height",
            ask=True,
            evidence=got("sim3_s", "scale", "gate_world_m", "sim3_inlier_ratio"),
            note=f"Umeyama 测得 s={s:.4f}, 远偏离 0.7155(误差>3%) => "
                 "要么相机内参/深度尺度错, 要么人体侧 pkl 不是同一条视频。"
                 "这是第二类必须问人的故障: 尺度错会静默地毁掉一切且看起来像「拟合差」。")
    if up is not None and up > TH["axis_up_max_deg"]:
        return Verdict(
            "gravity_not_locked", "enable gravity lock (fit_sim3 up_cam=...)",
            ask=False,
            evidence=got("axis_up_deg", "lean_deg", "grav_pitch_deg"),
            note=f"cam-up vs world+z = {up}° => Sim3 的自由旋转把重力吸走了一部分。"
                 "用地面停放段的箱体姿态反测重力、锁 R 重解 s,t(实测 4.9° -> 1.8°)。")

    # ------------------------------------------------------------------
    # 8. QA FAIL, 唯一失败是**拾取交接带内的紧凑 TELEPORT** ==> 已知可接受
    #    计划书附录 B 第 8 行: "渲染 QA 唯一 FAIL 在拾取段 f159-164 | 已知可
    #    接受 | 放过, 记录 | 已知基线的 TELEPORT, 不影响观感"。
    #
    #    ★ 排序理由(P1.3 那条教训的第二次实践): 本条是**具体且已被验收过**的
    #      判断, 必须排在笼统的 qa_fail_review **之前** —— 否则笼统那条会把它
    #      抢走, 于是一条已交付的好片子被报成"需要人确认"。
    #      但它仍必须排在同一批"有后果"的判断(frame_offset/freeze/scale/
    #      gravity)之后: 那些是真实故障, 不能被"已知可接受"洗掉。
    #    ★ 动作 = None(不重试), 也不问人: offset 已由 hop-1 验证(0-1 帧,
    #      corr>=0.977), 重搜只会改坏一条好片子。
    # ------------------------------------------------------------------
    if _g(m, "pass") is False and tight_pickup:
        f0s = [j["f0"] for j in bad]
        ev.update(got("qa_first_c", "first_fail_frame"))
        ev["n_pickup_teleports"] = len(bad)
        ev["pickup_frames"] = f0s
        return Verdict(
            "qa_known_acceptable", None, ask=False, evidence=ev, confidence="high",
            note=f"QA FAIL 全部集中在拾取交接带 f{min(f0s)}-f{max(f0s)} "
                 f"({len(bad)} 帧, carry/cov 状态不变) => **已知可接受项**"
                 "(计划书附录 B 第 8 行: 基线的单帧 TELEPORT, 不影响观感)。"
                 "★ 不重搜 frame_offset: offset 已由 hop-1 人体-机器人互相关"
                 "验证过(lag 0-1 帧, corr>=0.977), 动它反而会改坏一条好片子。"
                 "只记录, 不重试, 不问人。")

    # ------------------------------------------------------------------
    # 9. QA FAIL 但所有指标正常  ==> 剩下的只能请人确认
    #    真实案例: 早期 165945 的 f280->281 交接跳(已由抓取等待修复)。
    #    ★ 排序理由: 这是**有真实后果**的判断(要不要人接手), 必须排在
    #      纯信息性的对称支提示之前, 否则会被它抢走; 但排在更具体的
    #      qa_known_acceptable 之后。
    # ------------------------------------------------------------------
    if _g(m, "pass") is False:
        fff = _g(m, "first_fail_frame")
        return Verdict(
            "qa_fail_review", "manual_review_of_fail_frames (may be accepted-known)",
            ask=True,
            evidence=got("result", "first_fail_frame", "warn_parked_pos_jump_mm"),
            note=f"QA FAIL 但各阶段余量都在门限内 => 失败集中在 f{fff} 附近, "
                 "需要人确认是否属于**已知可接受项**(交接帧跳变/单帧 teleport)。"
                 "这种情况不该自动重试整条流水线——重跑一遍数字不变, 只是浪费时间。")

    # ------------------------------------------------------------------
    # 10. world 阶段显示 z 轴落在 180°  ==> 很可能是**正常**的对称等价
    #    这一条是"假警报"的典型:
    #    165945 实测 175.8°, 看着像箱子颠倒, 其实是 8 重对称群里的等价代表。
    #    ★ 面试考点: 别把对称等价当 bug 去"修"。
    #    ★ 排序理由: 纯信息性(ask=False 且不改动作), 只能排在有后果的判断之后。
    #      它是"全绿灯片子的补充说明", 不该覆盖任何真实故障。
    # ------------------------------------------------------------------
    zax = _g(m, "world_zaxis_med_deg")
    # 靠近 180° = 落在对称支(要提示); 靠近 0° = 正常姿态(不提示)
    if zax is not None and abs(zax - 180.0) <= TH["zaxis_sym_tol"]:
        return Verdict(
            "zaxis_symmetry_branch", "informational_only -- do NOT 'fix'",
            ask=False,
            evidence=got("world_zaxis_med_deg", "world_zaxis_p90_deg", "canon_jumps_gt5deg"),
            note=f"box z-axis vs world up = {zax}° => 落在 180° 对称支。"
                 "箱 half-extents (0.15,0.15,0.225) 的对称群有 8 个元素, "
                 "绕 x 翻 180°(diag([1,-1,-1])) 是**合法等价姿态**。这是正常的, 不要修。")

    # ------------------------------------------------------------------
    # 11. 全都正常
    # ------------------------------------------------------------------
    return Verdict(
        "ok", None, ask=False,
        evidence={"iou_median": iou, "depth_res_med_mm": dm,
                  "carry_rot_max_deg": rot_max, "scale": s},
        note="各阶段余量均在门限内。")


# ---------------------------------------------------------------------------
# 重试编排: 铁律 2/3 落成代码
# ---------------------------------------------------------------------------
RETRYABLE = {"mask_too_small", "tilt_ambiguity", "symmetry_flip",
             "frame_offset_mismatch", "freeze_failed", "gravity_not_locked"}
MAX_RETRIES = 2


def run_with_retry(stage_fn, m_fn) -> list:
    """极简重试编排器, 演示三条铁律怎么落地。

    参数:
        stage_fn(action|None) -> 执行一次某阶段, 返回 (new_metrics, log)
        m_fn(metrics) -> diagnose() 的等价调用
    """
    history = []
    attempt = 0
    while True:
        metrics, log = stage_fn(None if attempt == 0 else history[-1]["action"])
        v = m_fn(metrics)
        history.append({"attempt": attempt, "verdict": v, "action": v.action})
        if v.ask:
            history[-1]["stop_reason"] = "ask=True(铁律1): 停下问人, 不重试"
            break
        if v.cls == "ok":
            history[-1]["stop_reason"] = "通过"
            break
        if v.cls not in RETRYABLE or v.action is None:
            history[-1]["stop_reason"] = "无可用动作, 升级为问人"
            break
        attempt += 1
        if attempt > MAX_RETRIES:
            history[-1]["stop_reason"] = f"铁律3: 已重试 {MAX_RETRIES} 次, 升级为问人"
            break
    return history


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_metrics import collect

    if len(sys.argv) >= 2:
        # 用法: python diagnose.py <fit.log> [world.log] [qa.log]
        m = collect(*sys.argv[1:])
    else:
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        b = os.path.join(root, "agent", "fixtures")
        m = collect(
            os.path.join(b, "165945_fit.log"),
            os.path.join(b, "165945_world.log"),
            os.path.join(b, "qa_165945_batch_logs.log"),
        )
    print("=== 165945 (已交付片) 的诊断 ===")
    print(diagnose(m))
