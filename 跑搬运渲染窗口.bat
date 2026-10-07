@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
if not defined PY set PY=python

echo ============================================
echo  机器人搬运箱子渲染窗口（实测 6-DoF 路线）
echo ============================================
echo.
echo 换素材：把 data\xxx_g1.pkl 拖到本 bat 上，或用下面的菜单。
echo 窗口操作：左键拖=旋转，滚轮=缩放，右键拖=平移。关闭窗口即退出。
echo.

if not "%~1"=="" (
  set "PKL=%~1"
  goto :have_pkl
)

rem ---- 扫描 data\ 里所有带实测世界轨迹的素材 ----
set "N=0"
for %%F in ("data\*_g1.pkl") do (
  set "S=%%~nF"
  set "P=!S!"
  if "!S:~-3!"=="_g1" set "P=!S:~0,-3!"
  if exist "data\!P!_world_6dof.npz" (
    set /a N+=1
    set "OPT!N!=%%F"
    echo   [!N!] !S!  共 data\!P!_world_6dof.npz
  ) else (
    echo   [ ]  !S!  ^(无实测轨迹，会退化成合成轨迹^)
  )
)
echo.
if !N!==0 (
  set "PKL=example\20260922_162859\data\20260922_162859_g1.pkl"
  echo   没有带实测轨迹的素材，使用随包样例：!PKL!
) else if !N!==1 (
  call set "PKL=%%OPT1%%"
  echo   仅一个带实测轨迹的素材，自动选择：!PKL!
) else (
  set /p "SEL=请输入编号后回车（直接回车 = 1）: "
  if "!SEL!"=="" set "SEL=1"
  call set "PKL=%%OPT!SEL!%%"
)

:have_pkl
rem 去掉 pkl 结尾的 _g1 得到前缀：20260917_212026_g1 -> 20260917_212026
for %%F in ("%PKL%") do set "STEM=%%~nF"
set "PREFIX=%STEM%"
if "%STEM:~-3%"=="_g1" set "PREFIX=%STEM:~0,-3%"

rem 世界系箱子轨迹：<前缀>_world_6dof.npz，退而求其次 <前缀>_world.npz
set "BOXW="
if exist "data\%PREFIX%_world_6dof.npz" set "BOXW=data\%PREFIX%_world_6dof.npz"
if not defined BOXW if exist "data\%PREFIX%_world.npz" set "BOXW=data\%PREFIX%_world.npz"

echo.
echo 机器人素材：%PKL%
if defined BOXW (
  echo 箱子实测轨迹：%BOXW%
) else (
  echo [!] 没找到 data\%PREFIX%_world_6dof.npz
  echo     先跑第 6 步 align_box_to_world.py 生成它，否则会退化成合成轨迹。
  echo.
  choice /c YN /m "没找到实测轨迹，仍然继续渲染吗"
  if errorlevel 2 exit /b 1
)
echo.

if defined BOXW (
  "%PY%" scripts\vis_robot_with_box.py --robot_motion "%PKL%" --box_pose_world "%BOXW%" --dump_box_pose "data\%PREFIX%_boxpose_used.npz"
) else (
  "%PY%" scripts\vis_robot_with_box.py --robot_motion "%PKL%"
)
if errorlevel 1 (
  echo.
  echo [出错了] 把上面的信息截图发过来。
)
pause
