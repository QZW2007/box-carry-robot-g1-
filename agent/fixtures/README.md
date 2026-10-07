# fixtures/ —— 诊断测试用的**真实**日志证据

这些文件不是手写的，都是**实际运行**产生的日志，用来给 `diagnose.py` 的
回归测试提供证据（`test_diagnose.py` 的 A / C 段）。

**为什么需要单独一份 fixture**：`batch_logs/` 里的 `qa_..._165945...log`
是 **9/27 旧产物**时代的日志，而实际交付的是 **10/6 的 v29**。
两者不是同一次渲染，QA 结论也不同（旧 = PASS，交付 = FAIL）。
用错证据会让测试"绿得没有意义"。

## 文件

| 文件 | 来源 | 关键内容 |
|---|---|---|
| `qa_20260922_165945.log` | `check_render_quality.py --npz 数据/20260922_165945/20260922_165945_boxpose_used.npz`（md5 63473c15，即**交付的 v29**） | RESULT: FAIL；唯一失败 = f159–164 的 5 个 TELEPORT（52–58 mm，carry=1->1） |

## 重新生成（当交付产物变化时）

```powershell
$py = "python"
cd "."
& $py ".workbuddy\skills\gmr-carry-video-pipeline\scripts\check_render_quality.py" `
  --npz "数据\20260922_165945\20260922_165945_boxpose_used.npz" `
  *> ".workbuddy\agent\fixtures\qa_20260922_165945.log"
```

**怎么判断用的是不是交付产物**（这是本目录存在的原因）：

```powershell
# 交付的 boxpose = md5 63473c15（见 make_deliverables.py 的说明）
(Get-FileHash "数据\20260922_165945\20260922_165945_boxpose_used.npz" -Algorithm MD5).Hash
(Get-FileHash ".workbuddy\boxpose_v29.npz" -Algorithm MD5).Hash        # 应相同
(Get-FileHash "data\20260922_165945_boxpose_used.npz" -Algorithm MD5).Hash  # 5e591c14 = 旧的
```

## 三个已知差异（写下来避免下次再踩）

1. **`data/` 里的 165945 boxpose 不是交付产物**。交付的是
   `数据/20260922_165945/...` 与 `.workbuddy/boxpose_v29.npz`（md5 63473c15）；
   `data/` 里的仍是 9/27 的 v26 时代版本（md5 5e591c14）。
   详见 `.workbuddy/make_deliverables.py` 顶部注释（它记录了那次误覆盖事故）。
2. **交付片的 QA 是 FAIL，不是 PASS**。失败全部集中在拾取交接带
   f159–164（5 帧 TELEPORT）。计划书**附录 B 第 8 行**明确把它列为
   "已知可接受 —— 放过，记录"，所以 `diagnose` 应判 `qa_known_acceptable`
   并**不重试**（重搜 `frame_offset` 会改坏这条好片子）。
3. **`freeze_failed` 不能用 `rot_max` 判**。交付片 FLOOR `rot_max`=1.501°、
   STOOL `rot_max`=1.068°，但两者 `p90` 都是 **0.000**。按 `max>1°` 判会把
   一条已验收的好片子误报成故障；判"整段在动"必须看 **p90**。

## 相关的真跑证据（不在本目录，但测试会引用）

| 文件 | 说明 |
|---|---|
| `.workbuddy/p02/A6b.log` | **真跑** HSV 收窄 50%（`4,120,130:14`）→ fit_ok 4.5%、bbox 倒挂 |
| `.workbuddy/p1/BADSIZE.log` | **真跑** box_size 填错（0.20/0.30）→ fit_ok 5.9%、掩码正常 |
| `.workbuddy/p1/R1.log` | 放宽 S/V **且**补色相上限 → fit_ok **99%** |
| `.workbuddy/p1/R2.log` | 修复规则实际输出的 HSV → fit_ok **100%**、IoU 中位 0.898 |
| `.workbuddy/p1/RELAXED.log` | **反例**：只降 S/V 下限 → 仍只 8%，bbox 仍倒挂 |
| `.workbuddy/p1/SVONLY.log` | 只把色相上限补回 24 → 99%（说明主因是色相上限） |
