"""One-click launcher: start the server if it isn't up, then open the app.

    python launch.py              start if needed, open the interface
    python launch.py --stop       stop a server this launcher started
    python launch.py --status     is it running, and where
    python launch.py --install-startup    run it automatically at login
    python launch.py --remove-startup

A browser cannot start a local process, so an installed PWA can never bring its
own backend up.  This script is the missing half: double-clicking it (or
forest_ai.bat) does both, and doing so twice is harmless — it notices an
existing server and just opens the window.

The server is spawned detached and windowless, so closing the launcher does not
kill it.  Its output goes to .cache/server.log; without that a crash on startup
would leave nothing to look at.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent
PID_FILE = ROOT / ".cache" / "server.pid"
LOG_FILE = ROOT / ".cache" / "server.log"
DEFAULT_PORT = 8000
IS_WINDOWS = os.name == "nt"


def url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_up(port: int, timeout: float = 1.0) -> bool:
    """Is *our* server answering, as opposed to something else on the port?"""
    try:
        with urllib.request.urlopen(f"{url(port)}/api/config", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def port_owner(port: int) -> int | None:
    """PID listening on the port, so --stop works on a manually started server."""
    if not IS_WINDOWS:
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
             f"-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def pythonw() -> str:
    """The windowless interpreter, so no console box is left hanging around."""
    exe = pathlib.Path(sys.executable)
    if IS_WINDOWS:
        cand = exe.with_name("pythonw.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def start_server(port: int, wait: float = 90.0) -> bool:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if IS_WINDOWS:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"on port {port} ===\n")
        log.flush()
        proc = subprocess.Popen(
            [pythonw(), str(ROOT / "serve.py"), "--port", str(port)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{proc.pid}\n{port}\n", encoding="utf-8")

    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up(port):
            return True
        if proc.poll() is not None:      # died during startup
            return False
        time.sleep(0.4)
    return False


def read_pid_file() -> tuple[int | None, int]:
    try:
        parts = PID_FILE.read_text(encoding="utf-8").split()
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else DEFAULT_PORT
    except Exception:
        return None, DEFAULT_PORT


def kill(pid: int) -> bool:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=20)
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


# --------------------------------------------------------------- startup ---
def startup_shortcut() -> pathlib.Path:
    appdata = os.environ.get("APPDATA", "")
    return (pathlib.Path(appdata) / "Microsoft" / "Windows" / "Start Menu"
            / "Programs" / "Startup" / "forest_ai.lnk")


def install_startup(port: int) -> None:
    if not IS_WINDOWS:
        sys.exit("--install-startup is Windows-only")
    lnk = startup_shortcut()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    # --no-browser: at login start the server quietly; opening a window then
    # would be a surprise, and the PWA icon is what the user will click
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
          f"$s.TargetPath='{pythonw()}';"
          f"$s.Arguments='\"{ROOT / 'launch.py'}\" --no-browser --port {port}';"
          f"$s.WorkingDirectory='{ROOT}';"
          f"$s.IconLocation='{ROOT / 'web' / 'static' / 'icons' / 'forest_ai.ico'}';"
          f"$s.Description='Start the forest_ai server';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   check=True, capture_output=True, timeout=30)
    print(f"the server will now start at login\n  {lnk}")


def remove_startup() -> None:
    lnk = startup_shortcut()
    if lnk.exists():
        lnk.unlink()
        print(f"removed {lnk}")
    else:
        print("nothing to remove - it was not set to start at login")


# ------------------------------------------------------------------ main ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-browser", action="store_true",
                    help="start the server but do not open a window")
    ap.add_argument("--install-startup", action="store_true")
    ap.add_argument("--remove-startup", action="store_true")
    args = ap.parse_args()

    if args.install_startup:
        return install_startup(args.port)
    if args.remove_startup:
        return remove_startup()

    if args.status:
        pid, pport = read_pid_file()
        running = is_up(args.port)
        print(f"server on port {args.port}: {'running' if running else 'not running'}")
        if running:
            print(f"  {url(args.port)}")
            owner = port_owner(args.port)
            print(f"  pid {owner if owner else pid or 'unknown'}")
        print(f"  log: {LOG_FILE}")
        print(f"  starts at login: {'yes' if startup_shortcut().exists() else 'no'}")
        return

    if args.stop:
        if not is_up(args.port):
            print(f"nothing running on port {args.port}")
            return
        pid = port_owner(args.port) or read_pid_file()[0]
        if pid and kill(pid):
            for _ in range(20):
                if not is_up(args.port):
                    break
                time.sleep(0.3)
            PID_FILE.unlink(missing_ok=True)
            print(f"stopped the server on port {args.port}")
        else:
            print(f"could not stop pid {pid}; close it from Task Manager")
        return

    if is_up(args.port):
        print(f"server already running - opening {url(args.port)}")
    else:
        # something is listening but it is not us: starting would just fail to
        # bind, and the bind error buried in a log is a poor way to learn that
        squatter = port_owner(args.port)
        if squatter:
            sys.exit(f"port {args.port} is already used by another program "
                     f"(pid {squatter}).\n"
                     f"close it, or pick another port:\n"
                     f"  python launch.py --port {args.port + 1}")

        print(f"starting the server on port {args.port}...")
        if not start_server(args.port):
            tail = ""
            try:
                tail = "\n".join(LOG_FILE.read_text(encoding="utf-8",
                                                    errors="replace").splitlines()[-15:])
            except OSError:
                pass
            sys.exit(f"the server did not come up.\n"
                     f"log: {LOG_FILE}\n\n{tail}")
        print("ready")

    if not args.no_browser:
        webbrowser.open(url(args.port))


if __name__ == "__main__":
    main()
