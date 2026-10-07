# -*- coding: utf-8 -*-
"""Unit-test clip_naming.canonical_prefix against the real recorder layout.

Builds a throw-away tree and checks every resolution rule, including the
"free-form bag name inside rgbd_<ts>/" case the user is about to use
(9.22.1.bag, 9.22.2.bag, ...).
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import clip_naming as cn  # noqa: E402

root = tempfile.mkdtemp(prefix="clipname_")
try:
    cases = []

    # 1. free-form bag inside the recorder's rgbd_<ts> folder (the new convention)
    d = os.path.join(root, "video", "bag", "rgbd_20260922_180000")
    os.makedirs(d)
    open(os.path.join(d, "9.22.2.bag"), "w").close()
    cases.append((os.path.join(d, "9.22.2.bag"), "20260922_180000",
                  "free-form name + rgbd_ folder"))

    # 2. two bags in DIFFERENT rgbd_ folders -> different prefixes
    d = os.path.join(root, "video", "bag", "rgbd_20260922_183000")
    os.makedirs(d)
    open(os.path.join(d, "9.22.3.bag"), "w").close()
    cases.append((os.path.join(d, "9.22.3.bag"), "20260922_183000",
                  "second recording, same day"))

    # 3. the realsense.db3 the recorder actually writes
    d = os.path.join(root, "video", "bag", "rgbd_20260922_190000")
    os.makedirs(d)
    open(os.path.join(d, "realsense.db3"), "w").close()
    cases.append((os.path.join(d, "realsense.db3"), "20260922_190000",
                  "recorder default name"))

    # 4. folder renamed to something free-form -> metadata.json must decide
    d = os.path.join(root, "video", "bag", "我的第二次录制")
    os.makedirs(d)
    open(os.path.join(d, "盒test.db3"), "w").close()
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"recorded_at": "2026-09-22T20:15:07.123456+08:00"}, f)
    cases.append((os.path.join(d, "盒test.db3"), "20260922_201507",
                  "free-form folder -> metadata recorded_at"))

    # 5. already-canonical file names, incl. the legacy _unitree suffix
    d = os.path.join(root, "video")
    open(os.path.join(d, "20260922_161727.bag"), "w").close()
    cases.append((os.path.join(d, "20260922_161727.bag"), "20260922_161727",
                  "canonical file name"))
    open(os.path.join(d, "20260911_211441_unitree.bag"), "w").close()
    cases.append((os.path.join(d, "20260911_211441_unitree.bag"),
                  "20260911_211441", "legacy _unitree suffix"))

    # 6. nothing identifying it at all -> None (must be reported, not guessed)
    open(os.path.join(d, "9.22.9.bag"), "w").close()
    cases.append((os.path.join(d, "9.22.9.bag"), None,
                  "bare 9.22.9.bag, no folder/metadata"))

    # 7. explicit overrides win over everything
    cfg = os.path.join(root, "clips.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"9.22.9.bag": "20260922_235959",
                   "20260922_161727.bag": "20260922_000001"}, f)
    no_cfg = os.path.join(root, "no_such_clips.json")

    bad = 0
    print("--- pure resolution (no overrides) ---")
    for path, want, label in cases:
        got = cn.canonical_prefix(path, cfg=no_cfg)
        ok = got == want
        bad += 0 if ok else 1
        print("%s  %-46s want=%-16s got=%s"
              % ("ok  " if ok else "FAIL", label, want, got))

    print("--- configs/clips.json overrides ---")
    for path, want, label in ((os.path.join(d, "9.22.9.bag"), "20260922_235959",
                               "override rescues an unresolvable bag"),
                              (os.path.join(d, "20260922_161727.bag"),
                               "20260922_000001",
                               "override beats the canonical file name")):
        got = cn.canonical_prefix(path, cfg=cfg)
        ok = got == want
        bad += 0 if ok else 1
        print("%s  %-46s want=%-16s got=%s"
              % ("ok  " if ok else "FAIL", label, want, got))

    # find_bag round trip + duplicate-prefix detection
    globs = (os.path.join(root, "video", "**", "*.bag"),
             os.path.join(root, "video", "**", "*.db3"))
    for label, c in (("without overrides", no_cfg), ("with overrides", cfg)):
        mp, unmatched = cn.bag_map(globs=globs, cfg=c)
        if c is no_cfg:
            n_unmatched_plain = len(unmatched)
            unmatched_plain = list(unmatched)
        else:
            n_unmatched_ov = len(unmatched)
        print("\nbag_map (%s): %d resolved, %d unmatched"
              % (label, len(mp), len(unmatched)))
        for b, p in sorted(mp.items(), key=lambda kv: kv[1]):
            print("  %-16s <- %s" % (p, os.path.relpath(b, root)))
        for b in unmatched:
            print("  %-16s <- %s" % ("UNMATCHED", os.path.relpath(b, root)))
    if n_unmatched_plain != 1:
        bad += 1
        print("FAIL: expected 1 unmatched bag without overrides, got %d"
              % n_unmatched_plain)
    if n_unmatched_ov != 0:
        bad += 1
        print("FAIL: expected 0 unmatched with overrides, got %d"
              % n_unmatched_ov)

    got = cn.find_bag("20260922_180000", globs=globs, cfg=no_cfg)
    exp = os.path.join(root, "video", "bag", "rgbd_20260922_180000", "9.22.2.bag")
    ok = os.path.abspath(got or "") == os.path.abspath(exp)
    bad += 0 if ok else 1
    print("\n%s  find_bag('20260922_180000') -> %s"
          % ("ok  " if ok else "FAIL", got and os.path.relpath(got, root)))
    print("find_bag('does_not_exist') -> %s"
          % cn.find_bag("does_not_exist", globs=globs, cfg=no_cfg))

    print("\n%s" % ("ALL PASS" if bad == 0 else "%d FAILURES" % bad))
    sys.exit(1 if bad else 0)
finally:
    shutil.rmtree(root, ignore_errors=True)
