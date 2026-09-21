@echo off
REM === Phase 2 — build CUDA sm_61 (GTX 1080) du fork llama-cpp-turboquant ===
REM Usage: cmd /c build_cu61.bat [configure|build]
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "E:\oneplus\geniex_harness\npu-rtx\reference\llama-cpp-turboquant"

if "%1"=="build" goto build

:configure
cmake -B build-cu61 -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61 -DLLAMA_CURL=OFF
if errorlevel 1 exit /b 1
echo CONFIGURE_DONE
exit /b 0

:build
cmake --build build-cu61 -j 8
if errorlevel 1 exit /b 1
echo BUILD_DONE
exit /b 0
