import html
import json


TYPE_COLORS = {
    "PERSON": "#2563EB",
    "TEAM": "#0D9488",
    "DEPARTMENT": "#7C3AED",
    "PROJECT": "#D97706",
    "PRODUCT": "#DB2777",
    "ORG": "#059669",
    "UNKNOWN": "#64748B",
}


def build_graph_data(nodes: list[dict], edges: list[dict]) -> dict:
    """Normalize graph nodes and edges for interactive visualization."""
    normalized_nodes = []
    seen_ids = set()

    for node in nodes:
        node_id = node.get("id") or node.get("label") or node.get("name")
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        entity_type = node.get("entity_type", "UNKNOWN")
        normalized_nodes.append({
            "id": node_id,
            "label": node.get("label", node_id),
            "entity_type": entity_type,
            "aliases": node.get("aliases", []),
            "color": TYPE_COLORS.get(entity_type, TYPE_COLORS["UNKNOWN"]),
        })

    normalized_edges = []
    for edge in edges:
        src = edge.get("source") or edge.get("subject")
        tgt = edge.get("target") or edge.get("object")
        if not src or not tgt:
            continue
        for endpoint in (src, tgt):
            if endpoint not in seen_ids:
                seen_ids.add(endpoint)
                normalized_nodes.append({
                    "id": endpoint,
                    "label": endpoint,
                    "entity_type": "UNKNOWN",
                    "aliases": [],
                    "color": TYPE_COLORS["UNKNOWN"],
                })
        normalized_edges.append({
            "source": src,
            "target": tgt,
            "predicate": edge.get("predicate", "RELATED_TO"),
            "evidence": edge.get("evidence", ""),
        })

    return {"nodes": normalized_nodes, "edges": normalized_edges}


def render_knowledge_graph_html(
    nodes: list[dict],
    edges: list[dict],
    height: int = 520,
) -> str:
    """Generate a self-contained interactive HTML force-directed graph."""
    graph_data = build_graph_data(nodes, edges)
    payload_json = json.dumps(graph_data)
    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:12px;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;margin-right:5px;"></span>'
        f'{html.escape(label)}</span>'
        for label, color in TYPE_COLORS.items()
        if label != "UNKNOWN"
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #f8fafc; }}
  #toolbar {{ padding: 10px 14px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; }}
  #tooltip {{ position: fixed; pointer-events: none; background: rgba(15,23,42,0.95); border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; font-size: 12px; display: none; max-width: 300px; z-index: 10; }}
  canvas {{ display: block; width: 100%; height: {height - 44}px; cursor: grab; }}
</style>
</head>
<body>
<div id="toolbar">
  <div>{legend_items}</div>
  <div style="font-size:12px;color:#94a3b8;">Drag nodes to inspect | Hover for aliases &amp; evidence</div>
</div>
<div id="tooltip"></div>
<canvas id="kgCanvas"></canvas>
<script>
const data = {payload_json};
const canvas = document.getElementById("kgCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = {height - 44};
}}
resize();
window.addEventListener("resize", resize);

const nodes = data.nodes.map((n, i) => {{
  const angle = (2 * Math.PI * i) / Math.max(data.nodes.length, 1);
  const radius = Math.min(canvas.width, canvas.height) * 0.3;
  return {{
    ...n,
    x: canvas.width / 2 + radius * Math.cos(angle),
    y: canvas.height / 2 + radius * Math.sin(angle),
    vx: 0,
    vy: 0
  }};
}});
const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
const edges = data.edges.map(e => ({{
  ...e,
  sourceNode: nodeMap[e.source],
  targetNode: nodeMap[e.target]
}})).filter(e => e.sourceNode && e.targetNode);

let dragging = null;

canvas.addEventListener("mousedown", e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  dragging = nodes.find(n => Math.hypot(n.x - mx, n.y - my) < 22) || null;
}});
window.addEventListener("mouseup", () => dragging = null);
canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (dragging) {{
    dragging.x = mx;
    dragging.y = my;
    dragging.vx = 0;
    dragging.vy = 0;
  }}
  const hovered = nodes.find(n => Math.hypot(n.x - mx, n.y - my) < 22);
  if (hovered) {{
    const aliases = hovered.aliases && hovered.aliases.length ? `<br/><b>Aliases:</b> ${{hovered.aliases.join(", ")}}` : "";
    tooltip.innerHTML = `<b>${{hovered.label}}</b> (${{hovered.entity_type}})${{aliases}}`;
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top = (e.clientY + 12) + "px";
    tooltip.style.display = "block";
  }} else {{
    tooltip.style.display = "none";
  }}
}});

function tick() {{
  for (let i = 0; i < nodes.length; i++) {{
    for (let j = i + 1; j < nodes.length; j++) {{
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const dist = Math.max(Math.hypot(dx, dy), 1);
      const force = 2600 / (dist * dist);
      const fx = (dx / dist) * force, fy = (dy / dist) * force;
      nodes[i].vx -= fx; nodes[i].vy -= fy;
      nodes[j].vx += fx; nodes[j].vy += fy;
    }}
  }}
  for (const edge of edges) {{
    const dx = edge.targetNode.x - edge.sourceNode.x;
    const dy = edge.targetNode.y - edge.sourceNode.y;
    const dist = Math.max(Math.hypot(dx, dy), 1);
    const force = (dist - 150) * 0.02;
    const fx = (dx / dist) * force, fy = (dy / dist) * force;
    edge.sourceNode.vx += fx; edge.sourceNode.vy += fy;
    edge.targetNode.vx -= fx; edge.targetNode.vy -= fy;
  }}
  for (const n of nodes) {{
    if (n === dragging) continue;
    n.vx += (canvas.width / 2 - n.x) * 0.004;
    n.vy += (canvas.height / 2 - n.y) * 0.004;
    n.vx *= 0.85; n.vy *= 0.85;
    n.x = Math.max(40, Math.min(canvas.width - 40, n.x + n.vx));
    n.y = Math.max(30, Math.min(canvas.height - 30, n.y + n.vy));
  }}

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const edge of edges) {{
    const sx = edge.sourceNode.x, sy = edge.sourceNode.y;
    const tx = edge.targetNode.x, ty = edge.targetNode.y;
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();

    const angle = Math.atan2(ty - sy, tx - sx);
    const ax = tx - 22 * Math.cos(angle), ay = ty - 22 * Math.sin(angle);
    ctx.fillStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 8 * Math.cos(angle - 0.4), ay - 8 * Math.sin(angle - 0.4));
    ctx.lineTo(ax - 8 * Math.cos(angle + 0.4), ay - 8 * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fill();

    const mx = (sx + tx) / 2, my = (sy + ty) / 2;
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(edge.predicate, mx, my - 5);
  }}

  for (const n of nodes) {{
    ctx.fillStyle = n.color;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#f8fafc";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(n.label, n.x, n.y + 32);
  }}
  requestAnimationFrame(tick);
}}
tick();
</script>
</body>
</html>
"""