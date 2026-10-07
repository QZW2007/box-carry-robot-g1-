@echo off
set PY=python
if not defined PY set PY=python
cd /d "%~dp0"

echo ============================================
echo  GMR demo: record the retargeted motion to mp4
echo  A MuJoCo window opens AND records at the same time.
echo  Do NOT close the window or press Ctrl+C early --
echo  that leaves the mp4 unfinished and unplayable.
echo  Take used: 20260922_161727 (data\20260922_161727_g1.pkl)
echo ============================================
echo.

%PY% scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path data\20260922_161727_g1.pkl --record_video --video_path video\g1_demo.mp4

echo.
echo ============================================
echo  Done. Video should be at: video\g1_demo.mp4
echo  (Take used: 20260922_161727)
echo ============================================
pause
