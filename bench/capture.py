"""Reads the finished design out of Figma in one round trip, for scoring.

Deliberately separate from `critic.layout_script`: the gate only needs geometry
and runs on every step, while scoring runs once and needs token binding,
typography and spacing too. Keeping them apart means tuning one never silently
changes the other.

Sandbox rules apply here as much as anywhere: no optional chaining, and
`fontSize`/`fontName`/`fillStyleId` can all be `figma.mixed`, so every read is
type-checked before use.
"""
from __future__ import annotations

import json
from typing import Any

CAPTURE_SCRIPT = """\
const root = await figma.getNodeByIdAsync({node_id});
if (!root) {{ throw new Error('node not found'); }}

function paintInfo(node, info) {{
  if (!('fills' in node)) {{ return; }}
  const fills = node.fills;
  if (!Array.isArray(fills)) {{ return; }}
  for (const f of fills) {{
    if (f.type === 'SOLID' && f.visible !== false) {{
      info.hasSolidFill = true;
      if (f.boundVariables && f.boundVariables.color) {{ info.fillBound = true; }}
    }}
  }}
  // A paint style counts as token-backed: bootstrap_tokens binds those styles
  // to variables when it creates them.
  if ('fillStyleId' in node) {{
    const sid = node.fillStyleId;
    if (typeof sid === 'string' && sid.length > 0) {{ info.fillBound = true; }}
  }}
}}

function describe(node, depth) {{
  const info = {{
    id: node.id, name: node.name, type: node.type,
    x: Math.round(node.x), y: Math.round(node.y),
    width: Math.round(node.width), height: Math.round(node.height),
    visible: node.visible !== false,
    layoutMode: 'layoutMode' in node ? node.layoutMode : null,
    hasSolidFill: false, fillBound: false,
    children: []
  }};

  if ('itemSpacing' in node && typeof node.itemSpacing === 'number') {{
    info.itemSpacing = Math.round(node.itemSpacing);
  }}
  if ('paddingTop' in node) {{
    info.padding = [
      Math.round(node.paddingTop), Math.round(node.paddingRight),
      Math.round(node.paddingBottom), Math.round(node.paddingLeft)
    ];
  }}

  if (node.type === 'TEXT') {{
    info.characters = String(node.characters).slice(0, 160);
    info.fontSize = typeof node.fontSize === 'number' ? node.fontSize : null;
    if (node.fontName && typeof node.fontName === 'object' && node.fontName.family) {{
      info.fontFamily = node.fontName.family;
      info.fontStyle = node.fontName.style;
    }}
    const tsid = node.textStyleId;
    info.textStyled = typeof tsid === 'string' && tsid.length > 0;
  }}

  paintInfo(node, info);

  if ('children' in node && depth < {max_depth}) {{
    info.children = node.children.slice(0, 60).map(function (c) {{
      return describe(c, depth + 1);
    }});
  }}
  return info;
}}

return {{ createdNodeIds: [], tree: describe(root, 0) }};
"""

# Deeper than the visual gate's 4: scoring counts leaf text and buttons that
# sit several levels inside a card.
MAX_DEPTH = 8


def capture_script(node_id: str, max_depth: int = MAX_DEPTH) -> str:
    return CAPTURE_SCRIPT.format(node_id=json.dumps(node_id), max_depth=max_depth)


def walk(tree: dict) -> list[dict]:
    """Every node in the tree, root first. Scoring is all aggregate, so a flat
    list is easier to reason about than repeated recursion."""
    out: list[dict] = []
    stack: list[dict] = [tree]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.get("children") or [])
    return out


def capture(bridge: Any, node_id: str) -> dict | None:
    """Read the design for scoring, or None if the read failed."""
    from tools.figma_exec import execute_figma_js

    result = execute_figma_js(bridge, capture_script(node_id))
    if not result["ok"]:
        return None
    return (result.get("result") or {}).get("tree")
