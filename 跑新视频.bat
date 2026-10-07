@echo off
set PY=python
if not defined PY set PY=python
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  跑新视频：GVHMR 输出 -^> Unitree G1
echo.
echo  前提：先把 GVHMR 产出的 hmr4d_results.pt
echo        放到 data\ 下，命名成 data\视频名.pt
echo ============================================
echo.

if "%~1"=="" (
  set /p NAME=输入视频名（不带 .pt 扩展名，例如 myvideo）:
) else (
  set NAME=%~1
)

if not exist "data\%NAME%.pt" (
  echo.
  echo [错误] 找不到 data\%NAME%.pt
  echo        请先确认文件已放好，且名字对得上。
  pause
  exit /b 1
)

echo.
echo [1/2] .pt 转 SMPL-X npz ...
%PY% scripts\gvhmr_to_smplx.py --input "data\%NAME%.pt" --output "data\%NAME%_smplx.npz" --fps 30
if errorlevel 1 goto err

echo.
echo [2/2] 重定向到 Unitree G1（存 pkl + 录 mp4）...
%PY% scripts\run_retarget_headless.py --smplx_file "data\%NAME%_smplx.npz" --robot unitree_g1 --save_path "data\%NAME%_g1.pkl" --render_video "video\%NAME%_g1.mp4" --render_every 2
if errorlevel 1 goto err

echo.
echo ============================================
echo  完成！输出文件：
echo    data\%NAME%_g1.pkl     (机器人动作数据)
echo    video\%NAME%_g1.mp4    (可以直接双击播放)
echo ============================================
pause
exit /b 0

:err
echo.
echo ============================================
echo  出错了。把上面的信息截图发过来。
echo ============================================
pause
exit /b 1
