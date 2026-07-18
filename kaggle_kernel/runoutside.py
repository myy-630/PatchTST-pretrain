
import os
import subprocess
import sys
from pathlib import Path

# Kaggle Dataset 中的完整项目代码
PROJECT_DIR = Path(
    "/kaggle/input/datasets/"
    "meiyangyang630/codedataset"
)

# Kaggle Dataset 中的 SHDB-AF 数据
DATA_ROOT = Path(
    "/kaggle/input/datasets/"
    "meiyangyang630/ecgdatasetmyy/"
    "shd-af-clean-data"
)

# Kaggle 唯一允许长期写入训练输出的目录
OUTPUT_DIR = Path("/kaggle/working/outputs/pilot")

CONFIG_PATH = PROJECT_DIR / "configs" / "patchtst_ssl_pilot.yaml"
TRAIN_ENTRY = PROJECT_DIR / "run.py"


def check_path(name: str, path: Path) -> None:
    """检查指定路径是否存在。"""
    print(f"{name}: {path}")
    print(f"  exists: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(f"❌ {name} 不存在：{path}")


def install_dependencies() -> None:
    """安装 Kaggle 默认环境中缺少的项目依赖。"""
    print("\n" + "-" * 70)
    print("📦 检查并安装依赖")
    print("-" * 70)

    try:
        import wfdb

        print(f"✅ wfdb 已安装，版本：{wfdb.__version__}")
        return

    except ModuleNotFoundError:
        print("⚠️ 当前环境未检测到 wfdb，开始安装……")

    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "wfdb",
    ]

    print("执行安装命令：")
    print(" ".join(install_command))

    subprocess.run(
        install_command,
        check=True,
    )

    # 使用新的子进程验证，避免当前进程模块缓存影响
    verify_command = [
        sys.executable,
        "-c",
        (
            "import wfdb; "
            "print('✅ wfdb 安装成功，版本：', wfdb.__version__)"
        ),
    ]

    subprocess.run(
        verify_command,
        check=True,
    )


def main() -> None:
    print("=" * 70)
    print("🚀 PatchTST ECG Masked SSL — Kaggle Pilot")
    print("=" * 70)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Current working directory:", Path.cwd())
    print("-" * 70)

    # 1. 严格检查所有代码和数据路径
    check_path("PROJECT_DIR", PROJECT_DIR)
    check_path("DATA_ROOT", DATA_ROOT)
    check_path("TRAIN_ENTRY", TRAIN_ENTRY)
    check_path("CONFIG_PATH", CONFIG_PATH)
    check_path("train.csv", DATA_ROOT / "splits" / "train.csv")
    check_path("val.csv", DATA_ROOT / "splits" / "val.csv")
    check_path("raw_wfdb", DATA_ROOT / "raw_wfdb")

    # 2. 安装 Kaggle 环境中缺少的依赖
    install_dependencies()

    # 3. 创建输出目录
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. 将项目根目录注入 PYTHONPATH
    # 确保项目 run.py 内的 import src 能找到项目源码
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        str(PROJECT_DIR)
        if not existing_pythonpath
        else f"{PROJECT_DIR}{os.pathsep}{existing_pythonpath}"
    )

    # 减少 Python 输出缓冲，方便 Kaggle 实时显示训练日志
    env["PYTHONUNBUFFERED"] = "1"

    # 5. 构建启动命令
    command = [
        sys.executable,
        str(TRAIN_ENTRY),
        "--config",
        str(CONFIG_PATH),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUT_DIR),
    ]

    print("\n" + "-" * 70)
    print("▶️ 执行训练命令")
    print("-" * 70)
    print(" ".join(command))
    print("=" * 70, flush=True)

    # 6. 启动子进程训练
    # cwd=PROJECT_DIR，使项目 run.py 以项目根目录运行
    subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env=env,
        check=True,
    )

    # 7. 输出训练结果文件清单
    print("\n🎉 训练结束。输出文件清单：")

    has_outputs = False

    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            has_outputs = True
            size_mb = path.stat().st_size / 1024 / 1024

            print(
                f"  📂 {path.relative_to(OUTPUT_DIR)} "
                f"| {size_mb:.2f} MB"
            )

    if not has_outputs:
        print(
            "  ⚠️ 未检测到生成的输出文件，"
            "请检查项目 run.py 的保存路径和保存逻辑。"
        )


if __name__ == "__main__":
    main()

