-- 2D: Oxa as o3d's logo draws it, its own SVG (pixel art, 32 x 32), on a
-- pixel grid. It hops on the chips' kick and blinks; the gills swap poses
-- through the second bar while the riser climbs into the lift.

local score = require("track")
local ui = require("ui")

local SIZE = 512 -- 16 px a pixel: the lift's camera sees the voxels at this size
local X, Y = 960 - SIZE / 2, 540 - SIZE / 2

scene { "Title", start = score.time("title"), cue = "title", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  -- the grid of Oxa's pixels, reaching past it
  group({ opacity = 0.35 }, function()
    for i = -24, 24 do
      local a = (0.3 + 0.7 * progress(lb, 0, 2)) * (1 - math.abs(i) / 25)
      line { x1 = 960 + i * 16, y1 = 0, x2 = 960 + i * 16, y2 = 1080, stroke = ui.LINE, stroke_width = 1, opacity = a }
    end
    for j = -17, 17 do
      local a = (0.3 + 0.7 * progress(lb, 0, 2)) * (1 - math.abs(j) / 18)
      line { x1 = 0, y1 = 540 + j * 16, x2 = 1920, y2 = 540 + j * 16, stroke = ui.LINE, stroke_width = 1, opacity = a }
    end
  end)
  local hop = pulse(lb / 2, 5) -- the chips' kick: every other beat
  local enter = 0.85 + 0.15 * ui.soft(lb)
  local blink = (lb > 2.8 and lb < 3.1) or (lb > 5.8 and lb < 6.1)
  local gills = lb >= 4 and math.floor(lb) % 2 == 1
  group({ transform = { translate = { 0, -hop * 18 }, scale = { enter * (1 + 0.04 * hop), enter * (1 - 0.04 * hop) },
    origin = { 960, Y + SIZE } } }, function()
    svg { src = "oxa.svg", x = X, y = Y, w = SIZE, h = SIZE, overrides = {
      eyes_open = { visible = not blink }, eyes_closed = { visible = blink },
      gills_a = { visible = not gills }, gills_b = { visible = gills } } }
  end)
  group({ opacity = progress(lb, 1, 1.3) }, function()
    text { "assets/logo.svg · 32 × 32 pixels", x = 960, y = 880, align = "center", font = ui.MONO, size = 26, fill = ui.DIM }
  end)
end }
