"""Generate a rotating 3D neural network that folds, crosses and flips like a cone.

Model
    Layers are rings of nodes in the Y-Z plane, stacked along the X axis.
    Ring radii grow in sequence: layer 0 is the smallest (apex), the last layer
    is the largest (base). Colours follow the layer, not the size.

    One cycle (CYCLE = 4 * STEP_T seconds) is four moves, one starting every STEP_T:
      1. FLIP   radii morph: big rings shrink, small rings grow, until the order
                is reversed. Halfway, all rings have the same radius (a cylinder).
      2. PASS   apex-to-base distance L goes from +SWING to -SWING. Rings cross
                THROUGH each other; at L = 0 the cone is flat (concentric rings).
      3. FLIP   same morph, back to the original radii.
      4. PASS   L goes from -SWING to +SWING.
    A FLIP lasts FLIP_T and a PASS lasts PASS_T. Each uses a half-cosine ease
    (zero speed at both ends). If a move lasts longer than STEP_T, the next one
    starts while it is still slowing down, so the net never stops between moves.
    Overlap at a joint = duration of the move that ends there - STEP_T.
    With a duration equal to STEP_T the moves chain with a pause between.

    Ring l sits at a linear position between apex and base, around a pivot:
        x_l = L * (PIVOT - l / (n-1))
    PIVOT is the fixed point of the chain: 0 = apex still, 1 = base still,
    0.5 = middle ring still, apex and base moving in opposite directions.

    The whole net also rotates rigidly about a fixed axis (period T), seen
    through an orthographic camera.

Why sampling
    Rotation, axial motion and radius morphing have different timings, so a
    projected coordinate is not a single cosine. The script simulates N frames
    over one revolution and writes each attribute as ONE <animate values="...">.
    CYCLE must divide T so the loop closes exactly.

Usage:
    python3 scripts/gen_net.py            # uses ../assets relative to this file
    python3 scripts/gen_net.py <dir>      # dir containing header-dark/light.svg
"""
import math, re, sys, random
from pathlib import Path

# Default: the repo's assets/ folder, resolved from this file's location,
# so the script works regardless of the current working directory.
ASSETS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets"
for _n in ("dark", "light"):
    if not (ASSETS / f"header-{_n}.svg").is_file():
        sys.exit(f"error: {ASSETS / f'header-{_n}.svg'} not found. "
                 f"Pass the folder that contains the SVGs: python3 gen_net.py <dir>")

T = 18.0                       # seconds per revolution
BOX = (830, 34, 1165, 232)     # x0, y0, x1, y1 on the 1200x320 canvas
# (nodes, ring radius), apex -> base. Equal node counts: after the flip the
# smallest ring must not be crowded. Radii are linear, so mid-flip is a cylinder.
LAYERS = [(9, 0.30), (9, 0.52), (9, 0.74), (9, 0.96), (9, 1.18)]
K = 5                          # edges per node into the next ring
YAW, PITCH = math.radians(16), math.radians(14)   # camera orientation
AXIS_TILT = math.radians(5)    # rotation axis slightly off the cone axis

SWING = 2.2                    # apex-to-base distance when the cone is fully open
PIVOT = 0.5                    # 0 = apex fixed, 1 = base fixed, 0.5 = apex and base move oppositely
STEP_T = 4.5                   # s between the starts of consecutive moves
FLIP_T = 6.0                   # s the size inversion lasts
PASS_T = 6.0                   # s the rings take to cross each other
                               # Each must be in [STEP_T, 2*STEP_T]. Overlap at a joint =
                               # duration of the move that ends there - STEP_T.
                               # FLIP_T -> PASS_T joint overlaps FLIP_T - STEP_T;
                               # PASS_T -> FLIP_T joint overlaps PASS_T - STEP_T.
CYCLE = 4 * STEP_T             # s for the full sequence. Must divide T (18 s).
                               # STEP_T in {4.5, 2.25, 1.5, ...}
RING_OUTLINE = False           # True draws each ring as a thin ellipse through its nodes
N = 96                        # frames per revolution (linear interpolation)

assert abs(T / CYCLE - round(T / CYCLE)) < 1e-9, "CYCLE = 4*STEP_T must divide T"
for _name, _d in (("FLIP_T", FLIP_T), ("PASS_T", PASS_T)):
    assert STEP_T <= _d <= 2 * STEP_T, f"{_name} must be between STEP_T and 2*STEP_T"

random.seed(7)

def norm(v):
    n = math.sqrt(sum(c * c for c in v)); return [c / n for c in v]

def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def dot(a, b): return sum(x * y for x, y in zip(a, b))

# Nodes: ring l holds n_l nodes at angles theta (random start angle per ring).
n_l = len(LAYERS)
RADII = [r for _, r in LAYERS]
theta, layer_of = [], []
for li, (n, r) in enumerate(LAYERS):
    off = random.random() * 2 * math.pi
    for k in range(n):
        theta.append(off + 2 * math.pi * k / n)
        layer_of.append(li)

def adist(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)

# Connectivity by angular proximity, so strands stay radial in every pose:
# each node -> its K angularly nearest nodes in the next ring.
edges = set()
for i in range(len(theta)):
    nxt = [j for j in range(len(theta)) if layer_of[j] == layer_of[i] + 1]
    nxt.sort(key=lambda j: adist(theta[i], theta[j]))
    for j in nxt[:K]: edges.add((i, j))
for j in range(len(theta)):            # guarantee every node has an input
    if layer_of[j] == 0 or any(e[1] == j for e in edges): continue
    prev = [i for i in range(len(theta)) if layer_of[i] == layer_of[j] - 1]
    prev.sort(key=lambda i: adist(theta[i], theta[j]))
    edges.add((prev[0], j))
edges = sorted(edges)

# Rotation axis and camera basis. fwd points from the scene toward the camera,
# so a LARGER dot(p, fwd) means NEARER to the camera.
axis = norm([math.cos(AXIS_TILT), math.sin(AXIS_TILT), 0])
fwd = norm([math.sin(YAW) * math.cos(PITCH), math.sin(PITCH), math.cos(YAW) * math.cos(PITCH)])
right = norm(cross([0, 1, 0], fwd)); up = cross(fwd, right)

def view(p, ang):
    """Rotate p about `axis` by `ang` (Rodrigues), then project: (x, y_down, depth)."""
    c, s = math.cos(ang), math.sin(ang)
    k = dot(p, axis); par = [k * a for a in axis]
    w = cross(axis, p)
    q = [par[d] + (p[d] - par[d]) * c + w[d] * s for d in range(3)]
    return (dot(q, right), -dot(q, up), dot(q, fwd))

def ease(x):                           # half-cosine: 0 -> 1, zero slope at both ends
    return (1 - math.cos(math.pi * x)) / 2

def clamp01(x): return 0.0 if x < 0 else 1.0 if x > 1 else x

def pose(t):
    """(L, s) at time t. L = signed apex-to-base distance; s = radius mix, 0 = original
    radii, 1 = reversed radii.

    s = flip up (starts at 0) - flip down (starts at 2*STEP_T)
    q = cross 1 (starts at STEP_T) - cross 2 (starts at 3*STEP_T), L = SWING * (1 - 2q)
    The last move (cross 2) can outlast its cycle, so the neighbouring cycles
    k = -1, 0, +1 are summed. Completed moves cancel in pairs."""
    t += PASS_T - STEP_T               # t = 0 falls where cross 2 ends: cone fully open
    s = q = 0.0
    for k in (-1, 0, 1):
        b = t - k * CYCLE
        s += ease(clamp01(b / FLIP_T)) - ease(clamp01((b - 2 * STEP_T) / FLIP_T))
        q += ease(clamp01((b - STEP_T) / PASS_T)) - ease(clamp01((b - 3 * STEP_T) / PASS_T))
    return SWING * (1 - 2 * q), s

# Simulate one revolution. Frame N is frame 0, so the loop closes exactly.
# A node is  C + cos(theta) * A + sin(theta) * B,  and `view` is linear, so each
# ring is described by three projected vectors: centre C, axes A and B.
frames, rings = [], []                 # frames[k][i] = (x, y, z);  rings[k][l] = (C, A, B)
for k in range(N + 1):
    t = T * (k % N) / N
    ang = 2 * math.pi * t / T
    L, s = pose(t)                     # frame 0 = cone fully open, original radii
    rk = []
    for li in range(n_l):
        xl = L * (PIVOT - li / (n_l - 1))
        r = (1 - s) * RADII[li] + s * RADII[n_l - 1 - li]
        rk.append((view([xl, 0, 0], ang), view([0, r, 0], ang), view([0, 0, r], ang)))
    rings.append(rk)
    frames.append([tuple(rk[layer_of[i]][0][d]
                         + math.cos(theta[i]) * rk[layer_of[i]][1][d]
                         + math.sin(theta[i]) * rk[layer_of[i]][2][d] for d in range(3))
                   for i in range(len(theta))])

# Fit the whole animation envelope (rotation, motion and flip, ring ellipses
# included) into BOX.
allx = [p[0] for fr in frames for p in fr]
ally = [p[1] for fr in frames for p in fr]
for rk in rings:
    for C, A, B in rk:
        hx, hy = math.hypot(A[0], B[0]), math.hypot(A[1], B[1])
        allx += [C[0] - hx, C[0] + hx]; ally += [C[1] - hy, C[1] + hy]
allz = [p[2] for fr in frames for p in fr]
S = min((BOX[2] - BOX[0]) / (max(allx) - min(allx)), (BOX[3] - BOX[1]) / (max(ally) - min(ally)))
ox = (BOX[0] + BOX[2]) / 2 - S * (max(allx) + min(allx)) / 2
oy = (BOX[1] + BOX[3]) / 2 - S * (max(ally) + min(ally)) / 2
zmin, zmax = min(allz), max(allz)

def track(attr, vals, nd):
    """Static value from frame 0, plus one <animate> if the value ever changes."""
    strs = [f"{v:.{nd}f}" for v in vals]
    if len(set(strs)) == 1: return strs[0], ""
    return strs[0], (f'<animate attributeName="{attr}" dur="{T:g}s" repeatCount="indefinite" '
                     f'values="{";".join(strs)}"/>')

def sx(i): return [ox + S * fr[i][0] for fr in frames]
def sy(i): return [oy + S * fr[i][1] for fr in frames]
def depth(i, far, near):               # far -> `far`, near -> `near`
    return [far + (near - far) * (fr[i][2] - zmin) / (zmax - zmin) for fr in frames]

KAPPA = 0.5522847498            # cubic-Bezier constant for a quarter circle

def ring_d(rk, li):
    """Projected circle = affine image of a circle = ellipse; 4 cubic Beziers."""
    C, A, B = rk[li]
    cx, cy = ox + S * C[0], oy + S * C[1]
    ax, ay, bx, by = S * A[0], S * A[1], S * B[0], S * B[1]
    def P(sa, sb, ka=0.0, kb=0.0):     # c + (sa + ka) * a + (sb + kb) * b
        return f"{cx + (sa + ka) * ax + (sb + kb) * bx:.1f} {cy + (sa + ka) * ay + (sb + kb) * by:.1f}"
    k = KAPPA
    return (f"M{P(1,0)}C{P(1,0,0,k)} {P(0,1,k,0)} {P(0,1)}"
            f"C{P(0,1,-k,0)} {P(-1,0,0,k)} {P(-1,0)}"
            f"C{P(-1,0,0,-k)} {P(0,-1,-k,0)} {P(0,-1)}"
            f"C{P(0,-1,k,0)} {P(1,0,0,-k)} {P(1,0)}Z")

def lerp_hex(c1, c2, t):
    a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]; b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

def layer_color(li, pal):
    t = li / (n_l - 1)
    return lerp_hex(pal[0], pal[1], t * 2) if t <= .5 else lerp_hex(pal[1], pal[2], (t - .5) * 2)

def build(pal, edge_col, edge_op, ring_op):
    out = []
    if RING_OUTLINE:
        out.append(f'  <g fill="none" stroke-width="1.2" stroke-opacity="{ring_op}">')
        for li in range(n_l):
            ds = [ring_d(rk, li) for rk in rings]
            out.append(f'    <path stroke="{layer_color(li, pal)}" d="{ds[0]}">'
                       f'<animate attributeName="d" dur="{T:g}s" repeatCount="indefinite" '
                       f'values="{";".join(ds)}"/></path>')
        out.append("  </g>")
    out.append(f'  <g stroke="{edge_col}" stroke-opacity="{edge_op}" stroke-width="1">')
    for i, j in edges:
        a = [track("x1", sx(i), 1), track("y1", sy(i), 1), track("x2", sx(j), 1), track("y2", sy(j), 1)]
        out.append(f'    <line x1="{a[0][0]}" y1="{a[1][0]}" x2="{a[2][0]}" y2="{a[3][0]}">'
                   + "".join(s for _, s in a) + "</line>")
    out.append("  </g>")
    # Painter's order by mean depth: far nodes first, near nodes drawn on top.
    order = sorted(range(len(theta)), key=lambda i: sum(fr[i][2] for fr in frames))
    out.append("  <g>")
    for i in order:
        a = [track("cx", sx(i), 1), track("cy", sy(i), 1),
             track("r", depth(i, 2.4, 5.6), 2),             # near = larger
             track("fill-opacity", depth(i, 0.3, 1.0), 2)]  # near = brighter
        out.append(f'    <circle cx="{a[0][0]}" cy="{a[1][0]}" r="{a[2][0]}" '
                   f'fill-opacity="{a[3][0]}" fill="{layer_color(layer_of[i], pal)}">'
                   + "".join(s for _, s in a) + "</circle>")
    out.append("  </g>")
    return "\n".join(out)

THEMES = {   # palette, edge colour, edge opacity, ring opacity
    "dark":  (("#22d3ee", "#818cf8", "#f472b6"), "#8b949e", ".22", ".55"),
    "light": (("#0891b2", "#6366f1", "#db2777"), "#8c959f", ".38", ".6"),
}

for name, (pal, ec, eo, ro) in THEMES.items():
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
    net = (f"  <!-- net:start - 3D cone network, rotation {T:g}s, cycle {CYCLE:g}s, "
           f"{len(theta)} nodes, {len(edges)} edges -->\n"
           + build(pal, ec, eo, ro) + "\n  <!-- net:end -->")
    assert "<!--NET-->" in s, path
    s = s.replace("<!--NET-->", net)
    if "�" in s:
        sys.exit(f"error: {path} has invalid UTF-8 outside the network block. "
                 f"Restore the original SVG and run again.")
    path.write_bytes(s.encode("utf-8"))   # bytes: no locale encoding, no \r\n translation
    print(path, len(theta), "nodes", len(edges), "edges", len(s) // 1024, "KB")