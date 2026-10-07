@echo off

set PY=python

if not defined PY set PY=python

cd /d "%~dp0"



echo ============================================

echo  GMR demo: record the retargeted motion to mp4

echo  A MuJoCo window opens AND records at the same time.

echo  Do NOT close the window or press Ctrl+C early --

echo  that leaves the mp4 unfinished and unplayable.

echo  Take used: 20260922_162859 (example\20260922_162859\data\20260922_162859_g1.pkl)

echo ============================================

echo.



%PY% scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path example\20260922_162859\data\20260922_162859_g1.pkl --record_video --video_path example\20260922_162859\video\demo_retarget.mp4



echo.

echo ============================================

echo  Done. Video should be at: example\20260922_162859\video\demo_retarget.mp4

echo  (Take used: 20260922_162859)

echo ============================================

pause

