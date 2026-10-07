@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
if not defined PY set PY=python

echo ============================================
echo  重拍流程工具（详见 重拍指南.md）
echo ============================================
echo   [1] 生成打印材料（ArUco 标记 + 棋盘格）
echo   [2] 自检解算管线（无需任何素材）
echo   [3] 诊断照片：几张能检出棋盘格？
echo   [4] 标定相机（需先拍好棋盘格照片文件夹）
echo   [5] 解算箱体轨迹（需标定 json + 视频）
echo   [6] RealSense 深度直测一条龙（bag -> 轨迹 -> 对齐 -> 渲染）
echo ============================================
set /p CHOICE=输入编号后回车:

if "%CHOICE%"=="1" (
  %PY% scripts\generate_aruco_markers.py --ids 0 1 2 3 --size_mm 60 --checker 9 6 --square_mm 25 --out print_materials
  echo 材料在 print_materials\ 下。打印时选 100%% 实际大小，打印完用卷尺量！
  pause
) else if "%CHOICE%"=="2" (
  %PY% scripts\solve_box_pose.py --selftest
  pause
) else if "%CHOICE%"=="3" (
  set /p CALDIR=输入照片文件夹（回车默认 snapshot）:
  if "!CALDIR!"=="" set CALDIR=snapshot
  %PY% scripts\check_calib_shots.py --img_dir "!CALDIR!" --rows 9 --cols 6
  pause
) else if "%CHOICE%"=="4" (
  set /p CALDIR=输入照片文件夹（回车默认 calib_shots）:
  if "!CALDIR!"=="" set CALDIR=calib_shots
  %PY% scripts\calibrate_camera.py --img_dir "!CALDIR!" --rows 9 --cols 6 --square_mm 25 --out data\camera_calib.json
  pause
) else if "%CHOICE%"=="5" (
  set /p VID=输入视频路径（如 video\reshoot.mp4）:
  %PY% scripts\solve_box_pose.py --video "!VID!" --calib data\camera_calib.json --markers_json data\box_markers.json --out data\box_pose --preview video\box_pose_preview.mp4
  echo 先看 video\box_pose_preview.mp4 确认坐标轴钉在箱子上。
  pause
) else if "%CHOICE%"=="6" (
  set /p BAG=输入 bag 路径（回车默认 video\20260911_211441.bag）:
  if "!BAG!"=="" set BAG=video\20260911_211441.bag
  for %%F in ("!BAG!") do set BNAME=%%~nF
  set B2=!BNAME:_=!
  set /p PKL=输入机器人 pkl（回车默认 data\!B2!_unitree_g1.pkl）:
  if "!PKL!"=="" set PKL=data\!B2!_unitree_g1.pkl
  echo.
  echo [1/3] 深度提取箱体轨迹 ...
  %PY% scripts\box_traj_from_depth.py --bag "!BAG!" --out data\box_traj_depth.npz --preview video\box_depth_preview.mp4
  if errorlevel 1 goto rdfail
  echo.
  echo [2/3] 对齐到 G1 世界系（相机系 -^> 机器人系）...
  %PY% scripts\align_box_to_world.py --robot_motion "!PKL!" --box_traj data\box_traj_depth.npz --out data\box_traj_world.npz
  if errorlevel 1 goto rdfail
  echo 注意看本步输出：地面相位 z 的稳定度应 ^< 10 mm；若 toward-cam 偏 180 度，
  echo 在本步（第 2 步对齐）的上面那条命令后加 --flip_x 重跑本步
  echo （--flip_x 是 align_box_to_world.py 的参数，只属于第 2 步对齐，第 3 步渲染没有这个参数）。
  echo.
  echo [3^/3] 渲染实测轨迹视频 ...
  %PY% scripts\vis_robot_with_box.py --robot_motion "!PKL!" --box_traj_world data\box_traj_world.npz --record_video "video\!B2!_g1_measured_box.mp4"
  if errorlevel 1 goto rdfail
  echo.
  echo 完成！视频：video\!B2!_g1_measured_box.mp4
  pause
) else (
  echo 无效输入
  pause
)

:rdfail
echo.
echo [出错] 把上面的信息截图发过来。
pause
