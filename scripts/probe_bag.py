#!/usr/bin/env python
"""Pre-flight check for a RealSense bag: streams, resolution, fps, duration,
frame count and whether the exposure was LOCKED during recording.

Usage (gmr conda python, from the repo root):
    python scripts/probe_bag.py --bag video/20260922_161727.bag
    python scripts/probe_bag.py --bag video/xxx.bag --skip-frames

Why: a bag with auto-exposure left ON gives unstable depth and the 6-DoF fit
residual blows past 35 mm (happened on the 9.17 take).  Checking here costs
seconds instead of a wasted pipeline run.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--skip-frames", action="store_true",
                    help="report streams + duration only (no full decode)")
    args = ap.parse_args()

    if not os.path.exists(args.bag):
        sys.exit("bag not found: %s" % args.bag)

    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(args.bag, repeat_playback=False)
    prof = pipe.start(cfg)
    print("file  : %s  (%.0f MB)"
          % (args.bag, os.path.getsize(args.bag) / 1048576.0))

    print("streams:")
    for s in prof.get_streams():
        if s.is_video_stream_profile():
            v = s.as_video_stream_profile()
            print("  %-8s %4dx%d @ %2d fps  %s"
                  % (s.stream_name(), v.width(), v.height(), v.fps(),
                     s.format()))
        else:
            print("  %-8s %s" % (s.stream_name(), s.format()))

    playback = prof.get_device().as_playback()
    dur = playback.get_duration()
    secs = dur.total_seconds() if hasattr(dur, "total_seconds") else dur / 1e9
    print("duration: %.1f s" % secs)

    if not args.skip_frames:
        t0 = time.time()
        n_color = n_frames = 0
        first_color = None
        try:
            while True:
                fs = pipe.wait_for_frames()
                n_frames += 1
                c = fs.get_color_frame()
                if c:
                    n_color += 1
                    if first_color is None:
                        first_color = c
        except RuntimeError:
            pass
        print("framesets: %d   color frames: %d   (%.1f fps measured)"
              % (n_frames, n_color, n_color / max(secs, 1e-6)))
        print("decode time: %.1f s" % (time.time() - t0))
        if first_color is not None:
            for key in ("enable_auto_exposure", "exposure", "gain",
                        "enable_auto_white_balance", "white_balance"):
                try:
                    v = first_color.get_frame_metadata(
                        getattr(rs.frame_metadata_value, key))
                    print("  meta %-28s %s" % (key, v))
                except Exception:
                    pass
            locked = None
            try:
                locked = first_color.get_frame_metadata(
                    rs.frame_metadata_value.enable_auto_exposure) == 0
            except Exception:
                pass
            if locked is True:
                print("EXPOSURE: LOCKED (good)")
            elif locked is False:
                print("EXPOSURE: AUTO -- depth will drift, refit may fail "
                      "(the 9.17 take had this)")
    pipe.stop()


if __name__ == "__main__":
    main()
