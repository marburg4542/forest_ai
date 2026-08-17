"""Create Desktop and Start Menu shortcuts for the launcher (Windows).

    python tools/make_shortcuts.py            create both
    python tools/make_shortcuts.py --remove   delete them

The shortcuts point at forest_ai.bat, so clicking one starts the server if it
is not running and then opens the interface.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "forest_ai.bat"
ICON = ROOT / "web" / "static" / "icons" / "forest_ai.ico"
NAME = "forest_ai.lnk"


def locations() -> dict[str, pathlib.Path]:
    home = pathlib.Path(os.environ.get("USERPROFILE", pathlib.Path.home()))
    appdata = pathlib.Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    return {
        "Desktop": home / "Desktop" / NAME,
        "Start Menu": appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / NAME,
    }


def create(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # WindowStyle 7 = minimised: the batch console flashes as little as possible
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{path}');"
          f"$s.TargetPath='{TARGET}';"
          f"$s.WorkingDirectory='{ROOT}';"
          f"$s.IconLocation='{ICON}';"
          f"$s.WindowStyle=7;"
          f"$s.Description='forest_ai - tree extraction from point clouds';"
          f"$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   check=True, capture_output=True, timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    if os.name != "nt":
        sys.exit("this script is Windows-only")
    if not args.remove and not ICON.exists():
        sys.exit(f"icon missing: {ICON}\nrun: python tools/make_icons.py")

    for label, path in locations().items():
        if args.remove:
            if path.exists():
                path.unlink()
                print(f"  removed  {label}: {path}")
            else:
                print(f"  none     {label}")
        else:
            create(path)
            print(f"  created  {label}: {path}")


if __name__ == "__main__":
    main()
