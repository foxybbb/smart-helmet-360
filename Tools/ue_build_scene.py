"""Build the Smart Helmet test level in the running UE editor through UE's MCP server.

  /Game/SmartHelmet/Maps/HelmetTest  = copy of SimBlank (sky, sun) + ASmartHelmetTestRange
  + ASmartHelmetCharacter (operator with the 8-camera helmet) + GameMode override.
"""
import json
import sys

import ue_mcp

SCENE = "editor_toolset.toolsets.scene.SceneTools"
ASSET = "editor_toolset.toolsets.asset.AssetTools"
OBJ = "editor_toolset.toolsets.object.ObjectTools"
LEVEL = "/Game/SmartHelmet/Maps/HelmetTest"


def ts(toolset, tool, **args):
    r = ue_mcp.call("call_tool", {"toolset_name": toolset, "tool_name": tool, "arguments": args})
    print(f"{tool}: {str(r)[:300]}")
    return r


def ref(r):
    try:
        return json.loads(r)["returnValue"]["refPath"]
    except Exception:
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ue_mcp.connect()
    if "true" not in str(ts(ASSET, "exists", path=LEVEL)).lower():
        ts(ASSET, "create_folder", path="/Game/SmartHelmet/Maps")
        ts(ASSET, "duplicate", path="/Game/SimBlank/Levels/SimBlank", new_path=LEVEL)
    ts(SCENE, "load_level", level_path=LEVEL)

    existing = ts(SCENE, "find_actors", name="", tag="", collision_channels=[])
    if "SmartHelmetTestRange" not in existing:
        ts(SCENE, "add_to_scene_from_class", actor_type={"refPath": "/Script/MyProject.SmartHelmetTestRange"},
           name="HelmetTestRange", xform={"location": {"x": 0, "y": 0, "z": 0}})
    if "SmartHelmetCharacter" not in existing:
        ch = ts(SCENE, "add_to_scene_from_class", actor_type={"refPath": "/Script/MyProject.SmartHelmetCharacter"},
                name="HelmetOperator", xform={"location": {"x": 0, "y": 0, "z": 100}})
        p = ref(ch)
        if p:  # possess the placed operator when PIE starts
            ts(OBJ, "set_properties", instance={"refPath": p}, values=json.dumps({"AutoPossessPlayer": "Player0"}))

    ws = [a for a in json.loads(existing).get("returnValue", []) if "WorldSettings" in a["refPath"]]
    if ws:
        ts(OBJ, "set_properties", instance=ws[0],
           values=json.dumps({"DefaultGameMode": "/Script/MyProject.SmartHelmetGameMode"}))
    ts(ASSET, "save_assets", asset_paths=[])


if __name__ == "__main__":
    main()
