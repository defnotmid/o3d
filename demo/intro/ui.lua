-- Palette, type, springs and the small parts the scenes share. The colours
-- are Oxa's: its pinks, and its gills' red, green and blue (the X, Y and Z
-- axes).

local score = require("track")

local M = {}

M.BG, M.PANEL, M.LINE = "#0b0a12", "#16131f", "#2c2638"
M.TEXT, M.DIM = "#fff4f7", "#9a8fa6"
M.PINK, M.ROSE = "#ffb3c7", "#ff7fa3"
M.X, M.Y, M.Z = "#ff5a4e", "#4ee07a", "#4e9bff"

M.DISPLAY, M.SANS, M.MONO = "Bebas Neue", "Space Grotesk", "JetBrains Mono"

M.SNAP = spring { stiffness = 320, damping = 22 }
M.SOFT = spring { stiffness = 170, damping = 13 }
M.SWING = cubic_bezier(0.65, 0, 0.35, 1) -- camera moves

-- The song's beat length (one tempo throughout).
M.SPB = score.seconds("4n")

-- A spring released `beats` beats ago (0 before).
function M.snap(beats) return ease(M.SNAP, beats * M.SPB) end
function M.soft(beats) return ease(M.SOFT, beats * M.SPB) end

-- A scene's clock in beats.
function M.lb(f) return f.seconds / M.SPB end

-- A filled label, popping from its left edge; returns its width.
function M.chip(x, y, label, fill, ink, size, pop)
  size = size or 22
  local w = text_width { label, font = M.MONO, size = size } + size * 1.4
  local h = size * 1.9
  group({ transform = { scale = pop or 1, origin = { x, y + h / 2 } } }, function()
    rect { x = x, y = y, w = w, h = h, rx = h / 2, fill = fill }
    text { label, x = x + size * 0.7, y = y + h / 2, baseline = "middle", font = M.MONO, size = size, fill = ink or M.BG }
  end)
  return w
end

-- The section label top left: "02 · materials", popping on its first beat.
function M.label(lb, n, s)
  M.chip(160, 120, string.format("%02d · %s", n, s), M.LINE, M.TEXT, 24, M.snap(lb))
end

-- A fact line at the bottom: what, and where the number comes from.
function M.fact(lb, at, s, source)
  if lb < at then return end
  group({ opacity = progress(lb, at, at + 0.3), transform = translate(0, (1 - M.snap(lb - at)) * 30) }, function()
    text { s, x = 160, y = 960, font = M.SANS, size = 40, fill = M.TEXT }
    text { source, x = 160, y = 1004, font = M.MONO, size = 22, fill = M.DIM }
  end)
end

-- A caption under a 3D object: its three.js name and the fields that make
-- it, for a view's `labels` (bottom centre on the point).
function M.tag(at, title, detail, colour, pop)
  local w = math.max(text_width { title, font = M.MONO, size = 26 }, text_width { detail, font = M.MONO, size = 20 }) + 40
  return { at = at, w = w, h = 84, draw = function()
    group({ transform = { scale = pop, origin = { w / 2, 84 } }, opacity = clamp01(pop) }, function()
      rect { x = 0, y = 0, w = w, h = 84, rx = 14, fill = M.PANEL, fill_opacity = 0.9 }
      rect { x = 0, y = 0, w = 6, h = 84, rx = 3, fill = colour }
      text { title, x = w / 2, y = 36, align = "center", font = M.MONO, size = 26, fill = M.TEXT }
      text { detail, x = w / 2, y = 66, align = "center", font = M.MONO, size = 20, fill = M.DIM }
    end)
  end }
end

-- The soundtrack's spectrum at the frame, `n` bands from 0 to 1.
function M.spectrum(f, n)
  return audio_spectrum(f, { src = "score", sample_size = 512, smooth = 2, window = "hann", bands = n, log = true,
    levels = { -20, 26 } }) or {}
end

return M
