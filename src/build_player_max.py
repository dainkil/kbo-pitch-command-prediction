"""Build an aggressive Player-only probe from the best measured dFING2.

The only changed target term is dFING2's already LB-positive r13 role term.
q, temperature, PGR2, r12, and all other tables remain unchanged.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/workspace")
SRC = ROOT / "산출물" / "dain" / "src" / "submit_dFING2"
NAME = "submit_dFING2_PLAYER_MAX"
OUT = ROOT / "산출물" / "dain" / "src" / NAME
ZIP = ROOT / "산출물" / "dain" / "out" / f"{NAME}.zip"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    if ZIP.exists():
        ZIP.unlink()
    shutil.copytree(
        SRC,
        OUT,
        ignore=shutil.ignore_patterns("__pycache__", "output", "data", "*.zip", "*.bak*"),
    )
    meta_path = OUT / "model" / "jm_root" / "model" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old = float(meta.get("r13_g", 0.0))
    if old != 1.0:
        raise ValueError(f"unexpected dFING2 r13_g={old}")
    meta["r13_g"] = 1.25
    meta["player_max_parent"] = "dFING2"
    meta["player_max_change"] = "r13 role target offset scale 1.00 -> 1.25"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["7z", "a", "-tzip", "-mx=5", str(ZIP), ".",
         "-xr!__pycache__", "-xr!output", "-xr!data", "-xr!*.zip"],
        cwd=OUT, check=True, stdout=subprocess.DEVNULL,
    )
    print(ZIP)


if __name__ == "__main__":
    main()
