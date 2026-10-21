"""Set up dependencies in the project and launch the student demo."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import socket
import subprocess
import sys
import venv

from model_config import model_files, model_path

ROOT = Path(__file__).resolve().parent


def requirements_available() -> bool:
    try:
        from packaging.requirements import Requirement
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                requirement = Requirement(line)
                if version(requirement.name) not in requirement.specifier:
                    return False
        return True
    except (ImportError, PackageNotFoundError):
        return False


def main() -> int:
    if "--dependencies-ready" in sys.argv:
        return 0 if requirements_available() else 1
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise SystemExit("请安装64位Python 3.12（推荐）；支持的版本范围为3.12～3.14。")
    missing = [str(path) for path in model_files(model_path()) if not path.is_file()]
    if missing:
        raise SystemExit("缺少配套模型。请使用教师提供的完整版ZIP；若仓库已发布，也可从GitHub Releases获取。解压后再运行。\n缺少：\n" + "\n".join(missing))
    if requirements_available():
        python = Path(sys.executable)
    else:
        environment = ROOT / ".venv"
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.is_file():
            print("正在创建本项目的Python环境…", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        ready = subprocess.run([str(python), "-X", "utf8", str(ROOT / "bootstrap.py"), "--dependencies-ready"],
                               capture_output=True, cwd=ROOT)
        if ready.returncode:
            print("首次运行正在安装依赖，需要联网，请稍候…", flush=True)
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    if "--check" in sys.argv:
        print("环境检查通过。Python：", python, flush=True)
        return 0
    for port in range(8517, 8558):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        break
    else:
        raise SystemExit("8517～8557端口均被占用，请关闭旧演示后重试。")
    print(f"演示地址：http://127.0.0.1:{port}\n运行期间请保持此窗口打开。", flush=True)
    return subprocess.call([str(python), "-X", "utf8", "-m", "streamlit", "run", "app.py",
                            f"--server.headless={'true' if '--headless' in sys.argv else 'false'}",
                            f"--server.port={port}"], cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print("启动失败：", error, file=sys.stderr)
        raise SystemExit(1)
