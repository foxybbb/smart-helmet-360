"""Minimal CLI client for the Unreal Engine 5.8 MCP server (http://127.0.0.1:8000/mcp).

Usage:
  python ue_mcp.py tools                       # tools/list
  python ue_mcp.py call <tool> '<json args>'   # tools/call
  python ue_mcp.py ts <toolset> <tool> '<json>'  # call_tool shortcut (tool-search mode)
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/mcp"
_session = None
_id = 0


def rpc(method, params=None, notify=False):
    global _session, _id
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        _id += 1
        body["id"] = _id
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    req.add_header("MCP-Protocol-Version", "2025-06-18")
    if _session:
        req.add_header("Mcp-Session-Id", _session)
    with urllib.request.urlopen(req, timeout=600) as r:
        _session = r.headers.get("Mcp-Session-Id", _session)
        raw = r.read().decode("utf-8", "replace")
    if notify or not raw.strip():
        return None
    if raw.lstrip().startswith("event:") or raw.lstrip().startswith("data:"):
        raw = "\n".join(l[5:] for l in raw.splitlines() if l.startswith("data:"))
    return json.loads(raw)


def connect():
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "smart-helmet-cli", "version": "1"}})
    rpc("notifications/initialized", notify=True)


def call(tool, args):
    res = rpc("tools/call", {"name": tool, "arguments": args})
    if "error" in res:
        return res["error"]
    out = []
    for c in res["result"].get("content", []):
        out.append(c.get("text", f"<{c.get('type')} {c.get('mimeType', '')}>"))
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    connect()
    cmd = sys.argv[1]
    if cmd == "tools":
        for t in rpc("tools/list")["result"]["tools"]:
            print(t["name"], "-", t.get("description", "")[:150].replace("\n", " "))
    elif cmd == "call":
        print(call(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}))
    elif cmd == "ts":
        args = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        print(call("call_tool", {"toolset_name": sys.argv[2], "tool_name": sys.argv[3], "arguments": args}))
