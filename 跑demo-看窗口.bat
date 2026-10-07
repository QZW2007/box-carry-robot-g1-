@echo off

set PY=python

if not defined PY set PY=python

cd /d "%~dp0"



echo ============================================

echo  GMR demo: SMPL-X motion -^> Unitree G1

echo  A MuJoCo window will open and play the motion.

echo  Take used: 20260922_162859 (example\20260922_162859\data\20260922_162859_smplx.npz)

echo ============================================

echo.



%PY% scripts\smplx_to_robot.py --smplx_file example\20260922_162859\data\20260922_162859_smplx.npz --robot unitree_g1 --save_path example\20260922_162859\data\demo_smplx_g1.pkl



echo.

echo ============================================

echo  Retargeted motion written to example\20260922_162859\data\demo_smplx_g1.pkl

echo  (deliberately NOT a delivery artifact: this script

echo  OVERWRITES --save_path).

echo  Finished. If you saw an error above, copy it

echo  and send it over.

echo ============================================

pause

