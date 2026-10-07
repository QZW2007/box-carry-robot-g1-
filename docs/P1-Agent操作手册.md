# P1 操作手册：Agent MVP

> 配套：`项目掌握计划-v2.md` 的 P1 节（4–5 天）。
> **实测日期**：2026-10-07。本文所有数字都是**真实运行**结果。
> **生产代码零改动**：`scripts/cuboid_fit.py` 的 SHA256 与开工前一致
> （`5DB4B5A7…D52C46`），本次只新增/修改 `.workbuddy/agent/` 下的文件。

---

## 0. 先纠正计划书的两个过期标注

| 计划书写 | 实际（2026-10-07 实测） |
|---|---|
| P1.2 「`parse_metrics.py` 已完成，16 项回归断言一次跑通」 | 断言存在，但**默认读的 QA 日志是过期 fixture**，selftest **FAIL 2 项**（`result` 期望 FAIL 实得 PASS、`first_fail_frame` 期望 280 实得 None）。已修，见 §3 |
| P1.3 「`diagnose.py` **13/13 通过**」 | 实得 **12/13**，A 用例系统性失败。根因同上。修完并扩充后为 **19/19** |

**根因（一条，影响两处）**：`batch_logs/qa_20260922_165945_boxpose_used.log`
是 **9/27 旧产物**（`data/…boxpose_used.npz`，md5 `5e591c14`）时代的日志；
而**实际交付**的是 **10/6 的 v29**（`数据/…boxpose_used.npz` 与
`.workbuddy/boxpose_v29.npz`，md5 `63473c15`）。两者不是同一次渲染：

- 旧产物 QA = **PASS**（`None -- clean`）
- **交付产物 QA = FAIL**，唯一失败是 f159–164 的 5 个 TELEPORT

> **教训**：先确认"测试吃的是不是交付产物"。用错证据，测试绿得也没意义。
> 判定方法见 `.workbuddy/agent/fixtures/README.md`。

---

## 1. 环境

```powershell
$py = "python"
cd "."
$env:PYTHONIOENCODING = "utf-8"     # 否则中文输出在 PowerShell 里是乱码
```

bag 在仓库外：`<数据根>\rgbd_<前缀>\realsense.bag`。

---

## 2. 四条命令看 P1 是否正常

```powershell
& $py .workbuddy\agent\parse_metrics.py           # 日志 -> 指标（回归断言）
& $py .workbuddy\agent\test_diagnose.py           # 规则覆盖 19 条
& $py .workbuddy\agent\tools.py --selftest        # 5 个工具的边界与解析
& $py .workbuddy\agent\run_agent.py --verify      # P1.4 两条验收（离线, 秒级）
```

**当前状态：四条全绿**（exit 0）。

---

## 3. P1.2 的 5 个工具（`tools.py`）

设计原则（计划书 P1.1）：**一次调用 = 一个可独立失败的阶段**；返回值机器可读；
幂等；每一级都带自己的质量指标。

| 工具 | 包装的脚本 | 结构化的失败信号 |
|---|---|---|
| `calibrate_hsv` | `calib_hsv.py` + `configs/boxes.json` | 覆盖率 >40%(太宽) / <0.2%(太紧) |
| `fit_box_6dof` | `box_traj_from_depth.py --fit-cuboid` | `fit_ok_ratio`、IoU 分位、深度残差分位、**拒绝原因直方图**、`bbox_inverted` |
| `align_to_world` | `align_box_to_world.py` | `rise_err_mm_vs_stool`、`axis_up_deg`、`scale`、`n_carry` |
| `render_and_qa` | `vis_robot_with_box.py` + `check_render_quality.py` | PASS/FAIL + **首个失败帧号** |
| `diagnose` | 不调脚本 | `(分类, 建议动作, 是否问人)` |

**关键补充**（原 `parse_metrics.py` 缺的）—— 为区分两类"拟合率低"而新增：

```python
reject_reasons{}      # {"IoU # < #": 308, "mask too small": 22, "no blob accepted": 8}
n_reject_mask         # 与几何无关的掩码级拒绝数
n_reject_iou          # 纯轮廓拒绝数
bbox_inverted         # 掩码包围盒"倒挂"(上界<下界) = 掩码被切碎
frac_reject_mask      # 掩码级拒绝占比
```

**这些信号是判据的基础**，改 `parse_metrics.py` 时不要删。

用法示例：

```python
from tools import fit_box_6dof, diagnose_tool
r = fit_box_6dof(bag=r"<数据根>\rgbd_20260922_165945\realsense.bag",
                 hsv="4,75,79:24,255,255", out=".workbuddy/x_6dof.npz")
print(r["ok"], r["quality"])
print(diagnose_tool(r["metrics"])["cls"])
```

---

## 4. 修掉的三个真实缺陷（P1 的真正价值）

### 缺陷 1：`mask_too_small` 与 `metric_mismatch` 混淆 ★ 影响铁律 1

**症状**：HSV 收窄导致的掩码退化，被判成 `metric_mismatch`（**ask=True**，
"箱尺寸不符，停下问人"）——明明可以自动放宽 HSV 修好，却停下来问人。

**判别实验（两次真跑，完整 354 帧）**：

| 案例 | fit_ok | IoU 中位 | bbox_w_m | 倒挂 | 掩码级拒绝占比 |
|---|---|---|---|---|---|
| 健康基线 | 1.00 | 0.914 | [0.328, **0.52**] | — | — |
| **A6b HSV 收窄** | 0.045 | 0.606 | [0.284, **0.14**] | **True** | **8.9%** |
| **BADSIZE 尺寸错** | 0.059 | 0.579 | [0.328, 0.52] | False | **0.0%** |

**判据**：掩码"有病"（bbox 倒挂 或 存在掩码级拒绝）→ `mask_too_small`（可重试）；
掩码健康但轮廓对不上 → `metric_mismatch`（停下问人）。

> **面试说法**：「两类'拟合率低'长得几乎一样，但一个能自愈、一个必须问人。
> 区分它们靠的是**掩码本身的形态**：掩码被切碎时包围盒统计会倒挂
> （上界 0.14 < 下界 0.284），而且会出现 `mask too small` 这类与几何无关的拒绝；
> 尺寸填错时掩码是完好的，只是投影轮廓在任何位姿都对不上，
> 拒绝理由**全是** IoU 不够。」

### 缺陷 2：拾取段已知 TELEPORT 会触发重搜 offset ★ 会改坏好片子

**症状**：交付片 v29 的 QA 唯一失败是 f159–164 的 5 个 TELEPORT，
原规则判 `frame_offset_mismatch` 并**自动重搜 offset**——
而计划书**附录 B 第 8 行**明确写着这条是"已知可接受：放过，记录"。

**为什么要害**：`frame_offset` 已由 check_timing_sync 的 hop-1 验证过
（lag 0–1 帧，corr ≥ 0.977）。动它 = 改坏一个已验证正确的量。

**判据**：跳变**集中在拾取交接带**（first_c 之后、跨度 ≤12 帧、≤8 帧、
carry 状态不变）→ `qa_known_acceptable`（动作 None，不重试也不问人）；
否则才是 `frame_offset_mismatch`。

### 缺陷 3：`freeze_failed` 用 `rot_max` 判 → 误报已验收片子

**症状**：交付片 FLOOR `rot_max` = **1.501°**、STOOL `rot_max` = **1.068°**，
但两者 `p90` 都是 **0.000**。按 `max > 1°` 判会把一条已验收的好片子报成故障。

**判据**：判"整段在动"看 **p90**（门限 0.5° = QA 自己的 `PARKED_ROT_P90`）；
单帧 `max` 只是量化噪声。

> 这条和计划书 P1.3 里"告警门限 ≠ 故障门限"是**同一类错误**的第二次出现：
> 门限必须定在**噪声量级之上**，而且要用对**统计量**（p90 vs max）。

---

## 5. 铁律 2 的真实代价：只降 S/V 下限**修不好**

这是本次最有价值的一次实验。P1.3 写 `relax_hsv(-20 on S/V floor)`。我照做，实跑发现**没用**：

| HSV | fit_ok | bbox_w_m | 结论 |
|---|---|---|---|
| `4,120,130:**14**`（故障） | 4.5% | [0.284, **0.144**] 倒挂 | 坏 |
| `4,100,110:**14**`（只降 S/V） | **8%** | [0.285, **0.144**] **仍倒挂** | **没修好** |
| `4,120,130:**24**`（只补色相上限） | **99%** | [0.298, 0.499] | 基本修好 |
| `4,100,110:**24**`（两者都做） | **100%** | [0.316, 0.508] | **修好** |
| `0,100,110:**24**`（最终规则输出） | **100%** | [0.316, 0.508] IoU 0.898 | **修好** |

**结论**：故障主因是**色相上限被砍窄**（24→14 丢掉一半橙色像素），
S/V 下限只是次要因素。

**这正好是铁律 2 的深层含义**：重试不仅要"改变一件具体的事"，
还要**改到要害那件事**。只降 S/V 表面上是"一次具体的改变"，
但它改的是次要因素，所以照样修不好——**改了参数的无效重试仍然是无效重试**。
已把这个反例固化为测试 `C4`。

---

## 6. P1.4 的两条验收（真实运行，非模拟）

```powershell
# 验收 A: 给一个"HSV 标错"的输入, Agent 能自动放宽并跑通
& $py -u .workbuddy\agent\run_agent.py `
    --bag <数据根>\rgbd_20260922_165945\realsense.bag `
    --prefix E2E_HSV --hsv "4,120,130:14,255,255"

# 验收 B: 给一个"box_size 标错"的输入, Agent 能识别并停下问人
& $py -u .workbuddy\agent\run_agent.py `
    --bag <数据根>\rgbd_20260922_165945\realsense.bag `
    --prefix E2E_SIZE --hsv "4,75,79:24,255,255" --box-size 0.20,0.20,0.30
```

**验收 A 实测**（`.workbuddy/p1/agent_e2e_hsv.log`，耗时 32 分钟含两次全片拟合）：

| 轮 | HSV | fit_ok | IoU 中位 | 判定 | 动作 |
|---|---|---|---|---|---|
| 0 | `4,120,130:14` | **4.5%** | 0.606 | `mask_too_small` | 自动放宽 |
| 1 | `0,100,110:24` | **100%** | 0.898 | `ok` | 通过 |

**验收 B 实测**（`.workbuddy/p1/agent_e2e_size.log`）：

| 轮 | box_size | fit_ok | IoU 中位 | 判定 | 结果 |
|---|---|---|---|---|---|
| 0 | `0.20,0.20,0.30` | 5.9% | 0.579 | `metric_mismatch` | **ask=True，`n_attempts=1` 就停下** |

B 的关键数字是 `n_attempts = 1`：**没有浪费任何一次重试**。
这正是铁律 1 的意义——"知道什么时候停下来问人，是 Agent 成熟度的标志"。

---

## 7. 三条铁律在代码里的位置

| 铁律 | 位置 | 已验证 |
|---|---|---|
| 1. `ask=True` 绝不自动重试 | `run_agent.run_agent()` 首个分支；`RETRYABLE` 不含 ask 类 | ask 类 3 个分类都不在 `RETRYABLE`；验收 B 实测 `n_attempts=1` |
| 2. 重试必须改变一件具体的事 | `apply_action()` 返回 `changed`；`changed=False` 即停止 | 空动作/含糊动作 → `changed=False` → 停止 |
| 3. 重试上限 2 次 | `MAX_RETRIES = 2` | 早前一次真实冒烟跑了 3 次尝试后被截停 |

三者的常量**共用** `diagnose.py` 的 `RETRYABLE` / `MAX_RETRIES`，没有另立一套。

---

## 8. 产物索引

| 文件 | 内容 |
|---|---|
| `.workbuddy/agent/tools.py` | ★ P1.1/P1.2 的 5 个工具 |
| `.workbuddy/agent/run_agent.py` | ★ P1.4 编排器 + 三条铁律 + 验收脚本 |
| `.workbuddy/agent/diagnose.py` | 失败归因（11 条规则） |
| `.workbuddy/agent/parse_metrics.py` | 日志 -> 指标（含新增的掩码病灶信号） |
| `.workbuddy/agent/test_diagnose.py` | 19 条覆盖测试（A/B/C 三段） |
| `.workbuddy/agent/fixtures/` | ★ 由**交付产物**重生成的 QA 日志 + 三个已知差异说明 |
| `.workbuddy/p1/agent_e2e_hsv.log` | 验收 A 真实运行日志 |
| `.workbuddy/p1/agent_e2e_size.log` | 验收 B 真实运行日志 |
| `.workbuddy/p1/R2.log` | 修复规则输出 HSV 的实跑（100%） |
| `.workbuddy/p1/RELAXED.log` | ★ 反例：只降 S/V 修不好 |

---

## 9. 还没做的（诚实边界）

- **P1.5 的「人工干预 N 次」未测**：需要跑 3 批真实数据。
  按计划要求：**没测就不写比例**，只写机制。
- `tilt_ambiguity` 的 `densify_tilt_initialisers` 动作目前只改 `state` 参数，
  尚未接到 `cuboid_fit.initialisers()` 的实际入参上（该函数目前是硬编码的
  4×3×3 扫描）。真要用需给 `initialisers()` 加参数——**这是下一步**。

---

## 10. `--full` 下游链路：已接上并验证（2026-10-07 补）

上一版这里写的是"`--full` 未真跑"。**现已补完**，详见
**[下游链路验证报告.md](下游链路验证报告.md)**。

### 10.1 修了一个死参数

`--full` 之前**接进来了但从没被使用**（`run_agent()` 收了 `full` 参数却
在 `break` 前没做任何事）。现已实现 `run_downstream()`：

```
fit 判为 ok  →  align_to_world  →  render_and_qa  →  diagnose(整链视角)
```

**关键设计**：**必须等 `fit = ok` 才往下走**。带着一个 4% 有效率的 npz
去对齐+渲染，只会把 30 分钟花在注定失败的片子上，然后在最末端报一个
无法归因的 QA FAIL —— 那是"编排"而不是"归因"。

### 10.2 验证结果

**下游接线正确，且能复现交付结论**（`align` 与交付产物**逐位相同**）：

| 量 | 交付产物 | 下游链路 |
|---|---|---|
| `rise_err_mm_vs_stool` | +8 | **+8.0** |
| `axis_up_deg` | 1.8 | **1.8** |
| `scale` | 0.716 | **0.716** |
| `first_fail_frame` | 159 | **159** |
| 最终判定 | — | **`qa_known_acceptable`, ask=False** |

世界 npz 的 `box_center_world` / `box_R_world` 与交付对比 `max|diff| = 0`。

### 10.3 ★ 一个重要的诚实声明：交付的 v29 **无法逐字节重现**

用 `run_v29.ps1` 的**逐字参数**重跑 → 得到 md5 `b6b7108f`，
而交付的是 `63473c15`。差 70 帧的 `box_pos`（f145–f214，全在 carry 段），
中位 **22.9 mm**、max 34.9 mm。四元数**逐位相同**。

**根因**（逐层排除后锁定）：

```
v29 渲染完成          : 2026-10-06 14:51:09
vis_robot_with_box.py : 2026-10-06 16:19:40   ← 晚 89 分钟
```

渲染器在交付后被改过（手部穿模修复那批工作），改动经 **GRIP CLAMP**
传到 `box_pos`。交叉证据：`right_wrist_roll_link` 穿模帧数 v29 = **141**、
重跑 = **144**。

**已验证的排除项**（每一步都有证据，不是推测）：
输入 world npz 相同（全 0 差异）、命令行逐字相同、补了 `--hand_diag/--place_diag`
仍相同、用原始相对路径仍相同、**连跑两次 md5 相同（确定性）**。

**含义**：
1. 交付片本身没问题（QA 判定与重跑一致，失败项仍是已知可接受项）；
2. 但**已交付的 mp4/npz 是不可再生的快照** —— 日后"重跑一遍 v29"
   只会得到 `b6b7108f`；
3. 这是一个新的失败模式：**代码版本漂移**（输入全对、参数全对、引擎变了）。

### 10.4 为此新增了「脚本指纹」

`tools.fingerprints()` 把每个阶段脚本的 `sha256 + mtime` 随工具输出返回：

```python
r = fit_box_6dof(...)
r["fingerprint"]   # {"fit": {...}, "cuboid_fit": {...}}
```

本次就是靠 **mtime** 才定位到根因的（哈希/日志/参数都不露痕迹）。
以后比较两次运行的指纹，就能立刻区分"**参数变了**"与"**引擎变了**"。

### 10.5 复现命令

```powershell
# 下游链路验证（约 22 秒, 复用已交付 fit, 不重跑 10 分钟拟合）
& $py -u .workbuddy\agent\verify_full_downstream.py

# 完整 --full（约 30+ 分钟, 含 fit）
& $py -u .workbuddy\agent\run_agent.py `
    --bag <数据根>\rgbd_20260922_165945\realsense.bag `
    --robot data\20260922_165945_g1.pkl --prefix MYRUN `
    --hsv "4,75,79:24,255,255" --full
```
