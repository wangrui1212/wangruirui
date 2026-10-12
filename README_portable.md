# TDAS DAT Viewer 便携版

## 使用方法

1. 解压 `TDAS_DAT_Viewer_portable.zip`。
2. 进入解压后的 `TDAS_DAT_Viewer` 文件夹。
3. 双击 `TDAS_DAT_Viewer.exe` 启动程序。
4. 在界面中选择 `.dat` 文件或数据文件夹进行查看和分析。

请复制或移动整个 `TDAS_DAT_Viewer` 文件夹，不要只复制单个 `.exe` 文件。程序运行需要同目录下的依赖文件。

更新时请解压到新的空文件夹，不要覆盖解压到旧版本目录，以免遗留旧 DLL。

## 目标系统

- 推荐 Windows 10 / Windows 11 64 位。
- 目标电脑不需要安装 Python、PyQt6、numpy 或 pyqtgraph。
- 如果公司安全软件或 Windows Defender 对未签名 EXE 提示风险，请确认文件来源后放行。

## 重新构建

在项目根目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
```

构建成功后会生成：

- `dist\TDAS_DAT_Viewer\TDAS_DAT_Viewer.exe`
- `dist\TDAS_DAT_Viewer_portable.zip`

构建会隔离 PATH 中外部工具的 DLL 目录，避免误带入同名但不兼容的依赖；生成 ZIP 前会运行 `--smoke-test`，创建主窗口和实验报告工作台并检查正常退出。构建失败或启动检查失败会停止发布 ZIP。构建依赖已安装时，可传入 `-SkipDependencyInstall` 跳过安装。

构建脚本默认使用当前系统的 `python` 创建 `.venv-build` 虚拟环境，并安装 `requirements-build.txt` 中的打包依赖。如果当前 Python 版本与 PyInstaller 不兼容，建议改用 Python 3.13 x64 重新运行构建脚本。
