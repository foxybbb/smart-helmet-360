"""Copy the UE 5.8 template content the project uses (not stored in git: Epic template assets, ~130 MB).

  python Tools/setup_ue_assets.py ["C:/Program Files/Epic Games/UE_5.8"]

Copies TemplateResources/High/{Characters,Input,LevelPrototyping}/Content into MyProject/Content/.
The test level is then built through UE's MCP server:  python Tools/ue_build_scene.py
"""
import shutil
import sys
from pathlib import Path

ENGINE = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Program Files/Epic Games/UE_5.8")
CONTENT = Path(__file__).resolve().parents[1] / "MyProject" / "Content"
SRC = ENGINE / "Templates" / "TemplateResources" / "High"

for pack in ("Characters", "Input", "LevelPrototyping"):
    src, dst = SRC / pack / "Content", CONTENT / pack
    if not src.exists():
        sys.exit(f"not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("copied", pack, "->", dst)
print("SimBlank level: create MyProject from the 'Simulation / Blank' template or keep the existing Content/SimBlank.")
