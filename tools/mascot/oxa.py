# Oxa, the o3d mascot: a 32x32 pixel-art axolotl whose three pairs of gills
# are the axes, X red, Y green, Z blue. Writes an SVG whose parts are groups
# a script shows or hides by id:
#
#   gills_a      the gills (on)                gills_b      the gills waved
#   body         outline, head, body, tail, feet, cheeks (always on)
#   eyes_open    (on)                          eyes_closed  a happy blink
#   mouth_smile  (on)                          mouth_o      a little "o"
#
#   python3 tools/mascot/oxa.py assets/logo.svg   (also prints the body)
#
# Oxa is also the project's logo: render assets/logo.png (256 px) and
# assets/logo_32x32.png from it, e.g.
#   rsvg-convert -w 256 -h 256 assets/logo.svg -o assets/logo.png
import math, sys

W = H = 32
PAL = {
  'O': '#2b1b2e', 'B': '#ffb3c7', 'D': '#e88aa6', 'H': '#ffe0ea', 'L': '#ffd3df', 'C': '#ff7fa3',
  'E': '#1b1220', 'W': '#ffffff', 'M': '#7a1f2e',
  'x': '#ff5a4e', 'X': '#c43a3a', 'y': '#4ee07a', 'Y': '#2a9e52', 'z': '#4e9bff', 'Z': '#2f63c4',
}

def grid():
    return [['.'] * W for _ in range(H)]

def put(g, x, y, c):
    if 0 <= x < W and 0 <= y < H:
        g[y][x] = c

# A shape outlines itself over whatever is under it, then fills: a later
# shape overlapping an earlier one gets a dark seam (the chin over the body).
def shape(g, cells, fill):
    for (x, y) in cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) not in cells:
                put(g, x + dx, y + dy, 'O')
    for (x, y) in cells:
        put(g, x, y, fill(x, y))

def ellipse(cx, cy, rx, ry):
    return {(x, y) for y in range(H) for x in range(W)
            if ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1}

def shaded(cx, cy, rx, ry, belly=False):
    def fill(x, y):
        u, v = (x + 0.5 - cx) / rx, (y + 0.5 - cy) / ry
        if v > 0.62 or u * u + v * v > 0.8 and u + v > 0.5:
            return 'D'
        if u < -0.25 and v < -0.3 and u * u + v * v > 0.35:
            return 'H'
        if belly and abs(u) < 0.45 and v > -0.2:
            return 'L'
        return 'B'
    return fill

# A thick curve: the cells within radius r(t) of a quadratic bezier.
def stroke(p0, p1, p2, r0, r1):
    cells = set()
    for i in range(65):
        t = i / 64
        px = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        py = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        r = r0 + (r1 - r0) * t
        for y in range(H):
            for x in range(W):
                if math.hypot(x + 0.5 - px, y + 0.5 - py) <= r:
                    cells.add((x, y))
    return cells

# ---- the body: tail first, then body, feet and head over it ---------------------
body = grid()
shape(body, stroke((20, 26), (30, 28), (28.5, 17), 2.6, 1.0),
      lambda x, y: 'H' if y < 21 else 'D' if y > 25 else 'B')
shape(body, ellipse(15.5, 22.5, 7.0, 5.6), shaded(15.5, 22.5, 7.0, 5.6, belly=True))
for fx in (10, 19):
    shape(body, {(fx, 27), (fx + 1, 27), (fx, 28), (fx + 1, 28)}, lambda x, y: 'D' if y == 28 else 'B')
HX, HY, HRX, HRY = 15.5, 12.0, 9.8, 7.4
shape(body, ellipse(HX, HY, HRX, HRY), shaded(HX, HY, HRX, HRY))
for (x, y) in ((7, 14), (8, 14), (23, 14), (24, 14)):
    put(body, x, y, 'C')

# ---- the gills: three fronds a side, X on top, Y in the middle, Z below -----------
# Under the head (the group comes first), so the head covers their roots.
def gills(wave):
    g = grid()
    fronds = [((8, 8), (4, 4.5), (1.7, 1.7), 'x', 'X'), ((7, 11), (3.5, 10), (1.7, 7.5), 'y', 'Y'),
              ((7, 14), (3.5, 15), (1.7, 19), 'z', 'Z')]
    for root, mid, tip, light, dark in fronds:
        for side in (0, 1):
            m = lambda p: (p[0] if side == 0 else 32 - p[0], p[1] + (wave if side == 0 else -wave))
            r = m(root)[0], root[1]
            cells = stroke(r, m(mid), m(tip), 1.25, 0.9)
            shape(g, cells, lambda x, y, cy=root[1] + 0.5: light if y + 0.5 < cy + (1 if light == 'z' else 0) else dark)
    return g

# ---- the face -----------------------------------------------------------------------
def cells_layer(cells):
    g = grid()
    for (x, y, c) in cells:
        put(g, x, y, c)
    return g

eyes_open = cells_layer([(x, y, 'E') for x in (9, 10, 21, 22) for y in (11, 12)] +
                        [(9, 11, 'W'), (21, 11, 'W')])
eyes_closed = cells_layer([(x, y, 'E') for (x, y) in
                           ((8, 12), (9, 11), (10, 11), (11, 12), (20, 12), (21, 11), (22, 11), (23, 12))])
mouth_smile = cells_layer([(x, y, 'M') for (x, y) in
                           ((12, 15), (13, 16), (14, 16), (15, 17), (16, 17), (17, 16), (18, 16), (19, 15))])
mouth_o = cells_layer([(x, y, 'M') for x in (15, 16) for y in (16, 17)])

LAYERS = [('gills_a', gills(0), True), ('gills_b', gills(1), False), ('body', body, True),
          ('eyes_open', eyes_open, True), ('eyes_closed', eyes_closed, False),
          ('mouth_smile', mouth_smile, True), ('mouth_o', mouth_o, False)]

out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 8}" height="{H * 8}" shape-rendering="crispEdges">']
for name, g, on in LAYERS:
    rects = ''.join(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{PAL[g[y][x]]}"/>'
                    for y in range(H) for x in range(W) if g[y][x] != '.')
    out.append(f'<g id="{name}"{"" if on else " display=\"none\""}>{rects}</g>')
out.append('</svg>\n')
open(sys.argv[1], 'w').write('\n'.join(out))
print('\n'.join(''.join(r) for r in body))
