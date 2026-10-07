@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not defined PY set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ==================================================
echo  批量把 bag 转成 mp4
echo  bag 放在仓库外也支持（例如 <数据根>\rgbd_<时间戳>\realsense.bag）
echo  规范前缀 = 录制文件夹的时间戳，跟 bag 文件叫什么名字无关
echo  不会复制任何 bag，只在 video\ 下写出 <前缀>.mp4
echo ==================================================
echo.
set /p ROOT=输入 bag 所在根目录（回车默认 <数据根>）:
if "%ROOT%"=="" set ROOT=<数据根>

if not exist "%ROOT%" (
  echo [出错] 找不到目录 %ROOT%
  pause
  exit /b 1
)

echo.
echo ---- [1/3] 先看状态表（这一步不改任何东西）----
%PY% scripts\batch_pipeline.py --bag-root "%ROOT%"
if errorlevel 1 goto fail

echo.
echo ---- [2/3] 开始批量转 mp4 ----
echo （已经有 mp4 的会跳过；想全部重转就先删掉 video\ 下对应的 mp4）
%PY% scripts\batch_pipeline.py --bag-root "%ROOT%" --run --steps mp4
if errorlevel 1 goto fail

echo.
echo ---- [3/3] 转完后的状态（确认 mp4 列都是 Y）----
%PY% scripts\batch_pipeline.py --bag-root "%ROOT%"

echo.
echo 完成。mp4 在 video\<前缀>.mp4，可直接用播放器看。
echo.
echo 下一步（看上面的表）：
echo   * blocker 写 need-GVHMR(GPU) 的，说明缺 data\<前缀>.pt。
echo     去有 NVIDIA 显卡的机器跑 GVHMR，把 hmr4d_results.pt 改名拷回
echo     data\<前缀>.pt（表里 "bag -^> canonical prefix" 段落会提示名字）。
echo   * pt 都有了以后在本机跑全流程：
echo     %PY% scripts\batch_pipeline.py --bag-root "%ROOT%" --run
echo   * 想只跑某一步： --steps fit / world / render / sync
echo.
pause
exit /b 0

:fail
echo.
echo [出错] 把上面的信息截图发过来。
pause
