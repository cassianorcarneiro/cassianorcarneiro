"""
Generate a rotating 3D neural network that folds and crosses like a cone.

Model
    Layers are rings of nodes in the Y-Z plane, stacked along the X axis.
    Ring radii grow in sequence: layer 0 is the smallest (apex), the last layer
    is the largest (base). Radii are constant.

    The apex-to-base distance L oscillates between +SWING and -SWING. One cycle
    (CYCLE = 2 * (HOLD_T + PASS_T) seconds) has four parts:
      1. HOLD   rest with the cone fully open (L = +SWING), for HOLD_T.
      2. PASS   L goes from +SWING to -SWING in PASS_T. Rings cross THROUGH each
                other; at L = 0 the cone is flat (concentric rings).
      3. HOLD   rest with the cone inverted (L = -SWING), for HOLD_T.
      4. PASS   L goes from -SWING to +SWING.
    A PASS uses a half-cosine ease (zero speed at both ends). With HOLD_T = 0 the
    passes chain into a pure sinusoid, L = SWING * cos(pi * t / PASS_T).

    Ring l sits at a linear position between apex and base, around a pivot:
        x_l = L * (PIVOT - l / (n-1))
    PIVOT is the fixed point of the chain: 0 = apex still, 1 = base still,
    0.5 = middle ring still, apex and base moving in opposite directions.

    The whole net also rotates rigidly about a fixed axis (period T), seen
    through an orthographic camera.

Why sampling
    Rotation and axial motion have different timings, so a
    projected coordinate is not a single cosine. The script simulates N frames
    over one revolution and writes each attribute as ONE <animate values="...">.
    CYCLE must divide T so the loop closes exactly.

Canvas
    The script also resizes each SVG to HEIGHT (viewBox, height, the two background
    rects). The text block stays vertically centred and the wave stays at the
    bottom, both through a transform attribute. It stops with an error if the SVG
    header does not have the expected layout. Re-running is safe.
    SHOW_WAVE = False hides the two wave paths (display="none"). The paths stay in
    the file, so SHOW_WAVE = True brings them back on the next run.

Text
    The header strings are the HEADER_* constants below. The script writes them
    into the four <text> lines and the aria-label of both SVGs, XML-escaped.
    Position and font stay in the SVG (x, y and the .tag/.name/.sub/.tag2 CSS).
    The script warns if a line is long enough to reach the net.

Usage:
    python3 scripts/gen_net.py            # uses ../assets relative to this file
    python3 scripts/gen_net.py <dir>      # dir containing header-dark/light.svg
"""

# Imports

import math, re, sys, random
from pathlib import Path
from xml.sax.saxutils import escape

# Default: the repo's assets/ folder, resolved from this file's location,
# so the script works regardless of the current working directory.

ASSETS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets"

for _n in ("dark", "light"):
    if not (ASSETS / f"header-{_n}.svg").is_file():
        sys.exit(f"error: {ASSETS / f'header-{_n}.svg'} not found. "
                 f"Pass the folder that contains the SVGs: python3 gen_net.py <dir>")

# palette, edge colour, edge opacity, ring opacity

THEMES = {
    "dark":  (("#22d3ee", "#818cf8", "#f472b6"), "#8b949e", ".22", ".55"),
    "light": (("#0891b2", "#6366f1", "#db2777"), "#8c959f", ".38", ".6"),
}

# Total image height in px. Text stays centred and the wave stays at
# the bottom. The net's box grows with it (see BOX). If the net stops
# filling the width, adjust SWING: the script prints how much BOX is used.

HEIGHT = 400                   
assert HEIGHT >= 240, "HEIGHT too small for the net box"

# True: wave at the bottom. False: wave hidden. Without the wave, the bottom
# margin in BOX (86 px) is free: lower it to let the net use that space.

SHOW_WAVE = False

# x0, y0, x1, y1 of the net on the 1200 x HEIGHT canvas

BOX = (750, 28, 1162, HEIGHT - 86)

# Header text, written into both SVGs. Plain text: the script escapes & < > and quotes.

HEADER_TOP = "~/cassianorcarneiro"                                          # small cyan line above the name
HEADER_NAME = "Cassiano"                                                    # large title
HEADER_SUB = "Electrical Engineer (MSc) · Dev · ML Researcher"              # monospace subtitle
HEADER_TAG = ""                                                             # small grey line at the bottom
HEADER_ALT = "Cassiano — Electrical Engineer (MSc), Dev, ML Researcher"     # aria-label (screen readers)

# (nodes, ring radius), apex -> base.

LAYERS = [(9, 0.30), (9, 0.52), (9, 0.74), (9, 0.96), (9, 1.18)]

K = 10                                              # edges per node into the next ring
YAW, PITCH = math.radians(16), math.radians(14)     # camera orientation
AXIS_TILT = math.radians(5)                         # rotation axis slightly off the cone axis

PIVOT = 0.5                     # 0 = apex fixed, 1 = base fixed, 0.5 = apex and base move oppositely

REDUCTION = 2.0

T = 18.0 / REDUCTION            # seconds per revolution
SWING = 3.0                     # apex-to-base distance when the cone is fully open.
                                # Sets the length of the net. Ring diameter sets its height, so the fit is
                                # limited by one of the two. The script prints which, and how much of BOX is used.
PASS_T = 9.0 / REDUCTION        # s the rings take to cross each other
HOLD_T = 0.0 / REDUCTION        # s of rest at each extreme. 0 = continuous motion
CYCLE = 2 * (HOLD_T + PASS_T)   # s for the full sequence. Must divide T.
RING_OUTLINE = False            # True draws each ring as a thin ellipse through its nodes
N = 120                         # frames per revolution (linear interpolation)

assert PASS_T > 0 and HOLD_T >= 0, "PASS_T must be > 0 and HOLD_T >= 0"
assert abs(T / CYCLE - round(T / CYCLE)) < 1e-9, "CYCLE = 2*(HOLD_T+PASS_T) must divide T"

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

    """
    Rotate p about `axis` by `ang` (Rodrigues), then project: (x, y_down, depth).
    """

    c, s = math.cos(ang), math.sin(ang)
    k = dot(p, axis); par = [k * a for a in axis]
    w = cross(axis, p)
    q = [par[d] + (p[d] - par[d]) * c + w[d] * s for d in range(3)]
    return (dot(q, right), -dot(q, up), dot(q, fwd))

def pose(t):

    """
    Signed apex-to-base distance L at time t. t = 0 starts the rest with the cone open.
    SWING * cos(pi * x) is a half-cosine ease from +SWING to -SWING as x goes 0 -> 1.
    """

    c = t % CYCLE
    if c < HOLD_T:                     # 1. rest, cone open
        return SWING
    c -= HOLD_T
    if c < PASS_T:                     # 2. cross: +SWING -> -SWING
        return SWING * math.cos(math.pi * c / PASS_T)
    c -= PASS_T
    if c < HOLD_T:                     # 3. rest, cone inverted
        return -SWING
    c -= HOLD_T                        # 4. cross: -SWING -> +SWING
    return -SWING * math.cos(math.pi * c / PASS_T)

# Simulate one revolution. Frame N is frame 0, so the loop closes exactly.
# A node is  C + cos(theta) * A + sin(theta) * B,  and `view` is linear, so each
# ring is described by three projected vectors: centre C, axes A and B.

frames, rings = [], []                 # frames[k][i] = (x, y, z);  rings[k][l] = (C, A, B)

for k in range(N + 1):
    t = T * (k % N) / N
    ang = 2 * math.pi * t / T
    L = pose(t)                        # frame 0 = cone fully open
    rk = []
    for li in range(n_l):
        xl = L * (PIVOT - li / (n_l - 1))
        r = RADII[li]
        rk.append((view([xl, 0, 0], ang), view([0, r, 0], ang), view([0, 0, r], ang)))
    rings.append(rk)
    frames.append([tuple(rk[layer_of[i]][0][d]
                         + math.cos(theta[i]) * rk[layer_of[i]][1][d]
                         + math.sin(theta[i]) * rk[layer_of[i]][2][d] for d in range(3))
                   for i in range(len(theta))])

# Fit the whole animation envelope (rotation, motion, ring ellipses included) into BOX.

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
_w, _h = S * (max(allx) - min(allx)), S * (max(ally) - min(ally))
_bw, _bh = BOX[2] - BOX[0], BOX[3] - BOX[1]
print(f"fit: net {_w:.0f}x{_h:.0f} px in a {_bw}x{_bh} box "
      f"({'height' if _h * _bw > _w * _bh else 'width'} is the limit; "
      f"raise SWING to widen the net, lower it to narrow)")

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

    """
    Projected circle = affine image of a circle = ellipse; 4 cubic Beziers.
    """

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

def fit_canvas(s, path):
    """Resize the header canvas to HEIGHT. Sets absolute values, so it is safe to re-run
    and to go back to a smaller HEIGHT. The layout was designed at 320 px: the text
    group is shifted by half the extra height (stays centred) and the wave by all of it
    (stays at the same distance from the bottom)."""
    extra = HEIGHT - 320
    h = f"{HEIGHT:g}"
    s, n1 = re.subn(r'viewBox="0 0 1200 [\d.]+"', f'viewBox="0 0 1200 {h}"', s, count=1)
    s, n2 = re.subn(r'(<svg\b[^>]*?\sheight=")[\d.]+(")', rf'\g<1>{h}\g<2>', s, count=1)
    s, n3 = re.subn(r'(<rect width="1200" height=")[\d.]+(" fill="url\(#(?:bg|grid)\)"/>)',
                    rf'\g<1>{h}\g<2>', s)

    def place(s, classes, dy, show=True):
        def fix(m):
            tag = re.sub(r'\s+(?:transform|display)="[^"]*"', "", m.group(0))
            end = "/>" if tag.endswith("/>") else ">"
            extra = (f' transform="translate(0 {dy:g})"' if dy != 0 else "") + ("" if show else ' display="none"')
            return tag[:-len(end)] + extra + end
        return re.subn(r'<(?:g|path)\b[^>]*\bclass="(?:%s)"[^>]*>' % "|".join(classes), fix, s)

    s, n4 = place(s, ["fade"], extra / 2)            # text block
    s, n5 = place(s, ["wave", "wave2"], extra, SHOW_WAVE)   # the two wave paths
    if (n1, n2, n3, n4, n5) != (1, 1, 2, 1, 2):
        sys.exit(f"error: {path} does not have the expected header layout "
                 f"(viewBox, svg height, 2 background rects, 1 .fade group, 2 wave paths; "
                 f"found {n1}, {n2}, {n3}, {n4}, {n5}). Edit it by hand or restore the original.")
    return s

def set_text(s, path):

    """
    Write the HEADER_* strings into the four <text> lines and the SVG aria-label.
    Also warns when a line is probably long enough to reach the net.
    """

    lines = {"tag": HEADER_TOP, "name": HEADER_NAME, "sub": HEADER_SUB, "tag2": HEADER_TAG}
    for cls, text in lines.items():
        s, n = re.subn(r'(<text\b[^>]*\bclass="%s"[^>]*>)[^<]*(</text>)' % cls,
                       lambda m: m.group(1) + escape(text) + m.group(2), s)
        if n != 1:
            sys.exit(f'error: {path} should have exactly one <text class="{cls}">, found {n}. '
                     f"Edit it by hand or restore the original.")
        # Rough width check: monospace ~0.6 em per character, the bold sans title ~0.58 em.
        css = re.search(r'\.%s\{font:[^}]*\}' % cls, s)
        size = re.search(r'(\d+(?:\.\d+)?)px', css.group(0)) if css else None
        x = re.search(r'<text\b[^>]*\bx="([\d.]+)"[^>]*\bclass="%s"' % cls, s)
        if size and x:
            em = 0.6 if "monospace" in css.group(0) else 0.58
            end = float(x.group(1)) + len(text) * float(size.group(1)) * em
            if end > BOX[0] - 10:
                print(f'warning: {path.name}: line "{text}" ends near x={end:.0f}, '
                      f"close to the net (starts at x={BOX[0]}). Shorten it or move BOX.")
    alt = escape(HEADER_ALT, {'"': "&quot;"})
    s, n = re.subn(r'(<svg\b[^>]*\baria-label=")[^"]*(")', lambda m: m.group(1) + alt + m.group(2), s, count=1)
    if n != 1:
        sys.exit(f"error: {path} has no aria-label on the <svg> tag. Edit it by hand or restore the original.")
    return s

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
    s = fit_canvas(s, path)
    s = set_text(s, path)
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