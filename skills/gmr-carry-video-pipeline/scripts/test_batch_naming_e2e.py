# -*- coding: utf-8 -*-
"""End-to-end check of the free-form bag naming the user is about to use.

Reproduces exactly what happens when a fresh recording named `9.22.2.bag`
lands in `video/bag/rgbd_20260922_181000/` (the layout record_rgbd.py
creates), in three phases that mirror the real order of work:

  phase A -- straight off the recorder (no mp4, no data/<prefix>.pt):
    * the clip shows up under prefix 20260922_181000, derived from the FOLDER
      and not from the bag's own file name
    * its bag is found, so the row does not read "need-fit" with no bag
    * the blocker says `need-mp4`.  This is deliberate and was once a bug:
      mp4 needs no GPU, and --run skips anything reporting need-GVHMR, so if
      the GPU gate came first a fresh recording would never get converted and
      the user would have to run bag_to_mp4.py by hand for every bag.
    * the bag -> prefix map names the exact .pt to produce

  phase B -- after video/<prefix>.mp4 exists:
    * the blocker moves on to need-GVHMR(GPU), and the map still asks for the
      .pt (the GPU trip is deferred, not hidden)

  phase C -- after data/<prefix>.pt exists:
    * the blocker moves on to need-retarget

All phases run batch_pipeline.py as a subprocess with the repo root as cwd.
Everything it creates is removed again, including on failure.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
PY = os.environ.get("GMR_PYTHON", r"python")
PREFIX = "20260922_181000"
RDIR = os.path.join(ROOT, "video", "bag", "rgbd_%s" % PREFIX)
BAG = os.path.join(RDIR, "9.22.2.bag")
MP4 = os.path.join(ROOT, "video", "%s.mp4" % PREFIX)
PT = os.path.join(ROOT, "data", "%s.pt" % PREFIX)


def run_status():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([PY, "scripts/batch_pipeline.py"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    return r.stdout or ""


def row_for(out, blocker_word):
    return any(PREFIX in ln and blocker_word in ln for ln in out.splitlines())


def cells_for(out):
    """Whitespace-split cells of this clip's status-table row (None if absent).

    Used instead of substring matching so a check cannot pass by accident when
    the column widths shift.
    """
    for ln in out.splitlines():
        c = ln.split()
        if len(c) >= 5 and c[0] == PREFIX:
            return c
    return None


def check(label, ok):
    print("%s  %s" % ("ok  " if ok else "FAIL", label))
    return 0 if ok else 1


made_dir = False
made_mp4 = False
made_pt = False
bad = 0
try:
    os.makedirs(RDIR)
    made_dir = True
    with open(BAG, "wb") as f:            # a placeholder, not a real bag
        f.write(b"\x00" * 32)
    with open(os.path.join(RDIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"recorded_at": "2026-09-22T18:10:03.512764+08:00"}, f)

    # ---- phase A: fresh off the recorder (no mp4, no .pt) --------------
    out = run_status()
    print(out)
    bad += check("new prefix row exists", PREFIX in out)
    bad += check("resolved from the FOLDER, not the bag name",
                 "9.22.2.bag" in out and "-> %s" % PREFIX in out)
    bad += check("bag found for the new clip (not a bare need-fit row)",
                 any(PREFIX in ln and " Y " in ln for ln in out.splitlines()))
    bad += check("no mp4 yet -> need-mp4 (the GPU-free step runs first)",
                 row_for(out, "need-mp4"))
    bad += check("map tells the user which .pt to make",
                 "[need data/%s.pt]" % PREFIX in out)

    # ---- phase B: mp4 converted, GVHMR still not run -------------------
    with open(MP4, "wb") as f:
        f.write(b"\x00" * 32)
    made_mp4 = True
    out2 = run_status()
    bad += check("with the mp4 present -> blocked on the GPU step",
                 row_for(out2, "need-GVHMR"))
    bad += check("map still asks for the .pt (the GPU trip is not hidden)",
                 "[need data/%s.pt]" % PREFIX in out2)
    c2 = cells_for(out2) or []
    bad += check("bag and mp4 columns read Y, pt still '-'",
                 len(c2) >= 4 and c2[1] == "Y" and c2[2] == "Y" and c2[3] == "-")

    # ---- phase C: GVHMR output dropped in by hand ----------------------
    with open(PT, "wb") as f:
        f.write(b"\x00" * 32)
    made_pt = True
    out3 = run_status()
    bad += check("with data/%s.pt the row advances to need-retarget" % PREFIX,
                 row_for(out3, "need-retarget"))
    bad += check("map no longer asks for the .pt",
                 "[need data/%s.pt]" % PREFIX not in out3)

    print("\n%s" % ("ALL PASS" if bad == 0 else "%d FAILURES" % bad))
    sys.exit(1 if bad else 0)
finally:
    for made, path in ((made_pt, PT), (made_mp4, MP4)):
        if made and os.path.exists(path):
            os.remove(path)
    if made_dir and os.path.isdir(RDIR):
        shutil.rmtree(RDIR, ignore_errors=True)
    print("cleaned up: dir_gone=%s mp4_gone=%s pt_gone=%s"
          % (not os.path.exists(RDIR), not os.path.exists(MP4),
             not os.path.exists(PT)))
