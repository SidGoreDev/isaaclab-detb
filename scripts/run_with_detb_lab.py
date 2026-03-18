"""Bootstrap helper that exposes DETB Lab tasks to Isaac Lab scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Usage: run_with_detb_lab.py <script> [args...]")

    repo_root = Path(__file__).resolve().parents[1]
    extension_root = repo_root / "source" / "detb_lab"
    if str(extension_root) not in sys.path:
        sys.path.insert(0, str(extension_root))

    import detb_lab

    detb_lab.register_all()

    script_path = str(Path(args[0]).resolve())
    script_dir = str(Path(script_path).parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv = [script_path, *args[1:]]
    runpy.run_path(script_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
