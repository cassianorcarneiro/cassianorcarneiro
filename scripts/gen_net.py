"""
Generate a rotating 3D neural network for the README header SVGs.

Mechanism: rigid rotation about a fixed axis + orthographic camera.
Each projected coordinate is c + a*cos(wt) + b*sin(wt) = c + A*cos(wt - phi).
So every node/edge attribute is one SMIL <animate> with the same cosine
waveform (cubic keySplines), shifted by a negative `begin` = phase.

Usage:
    python3 scripts/gen_net.py            # uses ../assets relative to this file
    python3 scripts/gen_net.py <dir>      # dir containing header-dark/light.svg
"""

# Imports

import math, re, sys, random
from pathlib import Path

# Parameters

T = 20.0                                                                # seconds per revolution
BOX = (830, 34, 1165, 232)                                              # x0, y0, x1, y1 on the 1200x320 canvas
LAYERS = [(6, 0.42), (10, 0.70), (14, 0.86), (10, 0.70), (6, 0.42)]     # (nodes, ring radius)
K = 7                                                                   # edges per node into the next layer
YAW, PITCH = math.radians(64), math.radians(14)                         # camera orientation
AXIS_TILT = math.radians(7)                                             # rotation axis slightly off the layer axis

# Default: the repo's assets/ folder, resolved from this file's location,
# so the script works regardless of the current working directory.

ASSETS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets"
for _n in ("dark", "light"):
    if not (ASSETS / f"header-{_n}.svg").is_file():
        sys.exit(f"error: {ASSETS / f'header-{_n}.svg'} not found. "
                 f"Pass the folder that contains the SVGs: python3 gen_net.py <dir>")

# Bezier approximation of one quarter of cos(): handles at 1/3, end slope pi/2.

EI = "0.333 0 0.667 0.476"     # 1 -> 0 (accelerating)
EO = "0.333 0.524 0.667 1"     # 0 -> -1 (decelerating)
SPL = f"{EI};{EO};{EI};{EO}"

random.seed(7)

def norm(v):
    n = math.sqrt(sum(c * c for c in v)); return [c / n for c in v]

def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def dot(a, b): return sum(x * y for x, y in zip(a, b))

# 3D nodes: layers along X, each layer a ring in the Y-Z plane.
nodes, layer_of = [], []
n_l = len(LAYERS)
for li, (n, r) in enumerate(LAYERS):
    X = -1 + 2 * li / (n_l - 1)
    off = random.random() * 2 * math.pi
    for k in range(n):
        a = off + 2 * math.pi * k / n
        jitter = 1 + 0.08 * (random.random() - 0.5)
        nodes.append([X * 2.1, r * jitter * math.cos(a), r * jitter * math.sin(a)])
        layer_of.append(li)

# Sparse connectivity: each node -> K nearest nodes in the next layer.
edges = set()
for i, p in enumerate(nodes):
    nxt = [j for j, l in enumerate(layer_of) if l == layer_of[i] + 1]
    nxt.sort(key=lambda j: sum((p[d] - nodes[j][d]) ** 2 for d in range(3)))
    for j in nxt[:K]: edges.add((i, j))
for j, p in enumerate(nodes):          # guarantee every node has an input
    if layer_of[j] == 0 or any(e[1] == j for e in edges): continue
    prev = [i for i, l in enumerate(layer_of) if l == layer_of[j] - 1]
    prev.sort(key=lambda i: sum((p[d] - nodes[i][d]) ** 2 for d in range(3)))
    edges.add((prev[0], j))
edges = sorted(edges)

# Rotation axis and camera basis.
axis = norm([math.cos(AXIS_TILT), math.sin(AXIS_TILT), 0])
fwd = norm([math.sin(YAW) * math.cos(PITCH), math.sin(PITCH), math.cos(YAW) * math.cos(PITCH)])
right = norm(cross([0, 1, 0], fwd)); up = cross(fwd, right)

def decompose(p):
    """Rodrigues: R(t)p = par + perp*cos t + (axis x p)*sin t."""
    par = [dot(p, axis) * a for a in axis]
    perp = [p[d] - par[d] for d in range(3)]
    w = cross(axis, p)
    return par, perp, w

def coord(p, basis):
    par, perp, w = decompose(p)
    c, a, b = dot(par, basis), dot(perp, basis), dot(w, basis)
    return c, math.hypot(a, b), math.atan2(b, a)   # c + A cos(t - phi)

proj = [(coord(p, right), coord(p, up), coord(p, fwd)) for p in nodes]

# Fit the full orbit envelope into BOX (screen y grows downward).
xs = [c + s * A for (c, A, _), _, _ in proj for s in (-1, 1)]
ys = [-(c + s * A) for _, (c, A, _), _ in proj for s in (-1, 1)]
sx = (BOX[2] - BOX[0]) / (max(xs) - min(xs)); sy = (BOX[3] - BOX[1]) / (max(ys) - min(ys))
S = min(sx, sy)
cx0 = (BOX[0] + BOX[2]) / 2 - S * (max(xs) + min(xs)) / 2
cy0 = (BOX[1] + BOX[3]) / 2 - S * (max(ys) + min(ys)) / 2
zs = [c + s * A for _, _, (c, A, _) in proj for s in (-1, 1)]
zmin, zmax = min(zs), max(zs)

def f(v): return f"{v:.1f}".rstrip("0").rstrip(".")

def anim(attr, c, A, phi):
    """Static value at t=0, plus a cosine SMIL animation if it moves."""
    base = c + A * math.cos(-phi)
    if A < 0.05: return f(base), ""
    begin = ((phi % (2 * math.pi)) / (2 * math.pi)) * T - T
    vals = ";".join(f(v) for v in (c + A, c, c - A, c, c + A))
    return f(base), (f'<animate attributeName="{attr}" dur="{T:g}s" begin="{begin:.2f}s" '
                     f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.25;.5;.75;1" '
                     f'keySplines="{SPL}" values="{vals}"/>')

def screen(i):
    (xc, xa, xp), (yc, ya, yp), _ = proj[i]
    return (cx0 + S * xc, S * xa, xp), (cy0 - S * yc, S * ya, yp + math.pi)

def depth_map(i, lo, hi):
    zc, za, zp = proj[i][2]
    k = (hi - lo) / (zmax - zmin)
    return lo + k * (zc - zmin), k * za, zp

def lerp_hex(c1, c2, t):
    a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]; b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

def layer_color(li, pal):
    t = li / (n_l - 1)
    return lerp_hex(pal[0], pal[1], t * 2) if t <= .5 else lerp_hex(pal[1], pal[2], (t - .5) * 2)

def build(pal, edge_col, edge_op):
    out = [f'  <g stroke="{edge_col}" stroke-opacity="{edge_op}" stroke-width="1">']
    for i, j in edges:
        (x1, y1), (x2, y2) = screen(i), screen(j)
        a = [anim("x1", *x1), anim("y1", *y1), anim("x2", *x2), anim("y2", *y2)]
        out.append(f'    <line x1="{a[0][0]}" y1="{a[1][0]}" x2="{a[2][0]}" y2="{a[3][0]}">'
                   + "".join(s for _, s in a) + "</line>")
    out.append("  </g>")
    # Draw far-on-average nodes first: a static approximation of the painter's algorithm.
    order = sorted(range(len(nodes)), key=lambda i: -proj[i][2][0])
    out.append("  <g>")
    for i in order:
        x, y = screen(i)
        a = [anim("cx", *x), anim("cy", *y),
             anim("r", *depth_map(i, 5.6, 2.4)),            # near = larger
             anim("fill-opacity", *depth_map(i, 1.0, 0.3))]  # near = brighter
        out.append(f'    <circle cx="{a[0][0]}" cy="{a[1][0]}" r="{a[2][0]}" '
                   f'fill-opacity="{a[3][0]}" fill="{layer_color(layer_of[i], pal)}">'
                   + "".join(s for _, s in a) + "</circle>")
    out.append("  </g>")
    return "\n".join(out)

THEMES = {
    "dark":  (("#22d3ee", "#818cf8", "#f472b6"), "#8b949e", ".22"),
    "light": (("#0891b2", "#6366f1", "#db2777"), "#8c959f", ".38"),
}

for name, (pal, ec, eo) in THEMES.items():
    path = ASSETS / f"header-{name}.svg"
    # Explicit UTF-8: on Windows the default is cp1252, which writes "·" as the
    # single byte 0xB7 and produces invalid UTF-8 (XML parse error).
    # errors="replace" lets the script repair files corrupted by older versions:
    # the bad byte lives only inside the net block, which is regenerated below.
    s = path.read_bytes().decode("utf-8", errors="replace")
    # Git on Windows (core.autocrlf=true) checks files out with CRLF. The regexes
    # below match "\n", so normalize first. SVG/XML accepts LF everywhere.
    s = s.replace("\r\n", "\n")
    # Drop the old 2D network: CSS for .edge/.node and the two <g> blocks.
    s = re.sub(r"\s*\.edge\{[^}]*\}", "", s)
    s = re.sub(r"\s*\.node\{[^}]*\}", "", s)
    s = re.sub(r"\s*@keyframes pulse\{[^\n]*", "", s)
    s = s.replace(".wave,.wave2,.fade,.node{animation:none}", ".wave,.wave2,.fade{animation:none}")
    s = re.sub(r"\n  <g>\n    <line class=\"edge\".*?</g>\n  <g>\n    <circle class=\"node\".*?</g>\n",
               "\n<!--NET-->\n", s, flags=re.S)
    s = re.sub(r"\n  <!-- net:start.*?<!-- net:end -->\n", "\n<!--NET-->\n", s, flags=re.S)
    net = (f"  <!-- net:start - 3D network, rotation {T:g}s, {len(nodes)} nodes, {len(edges)} edges -->\n"
           + build(pal, ec, eo) + "\n  <!-- net:end -->")
    assert "<!--NET-->" in s, path
    s = s.replace("<!--NET-->", net)
    if "�" in s:
        sys.exit(f"error: {path} has invalid UTF-8 outside the network block. "
                 f"Restore the original SVG and run again.")
    path.write_bytes(s.encode("utf-8"))   # bytes: no locale encoding, no \r\n translation
    print(path, len(nodes), "nodes", len(edges), "edges", len(s) // 1024, "KB")