-- Two renderers, one picture, under the hook played in unison by a chip
-- and the strings. The left half is this video's own view, drawn on the
-- GPU; the right half is media/parity_software.mp4, the same view drawn by
-- o3d's software renderer (software.lua, `--software`), playing on the
-- same clock. The numbers are tools/gpu_check's (facts.lua).

local score = require("track")
local ui = require("ui")
local T = require("timeline")
local facts = require("facts")
local view = require("parity_view")

local CLIP = "parity_software.mp4"
local HAS_CLIP = io.open(SCRIPT_DIR .. "/media/" .. CLIP) ~= nil

scene { "Parity", start = score.time("parity"), cue = "parity", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  view(f)
  if HAS_CLIP then
    group({ clip = { x = 960, y = 0, w = 960, h = 1080 } }, function()
      video_frame(f, { src = CLIP, x = 0, y = 0, w = 1920, h = 1080 })
    end)
  end
  local split = ease(easing.exponential_out, lb, 0.6)
  if split > 0 then
    rect { x = 958, y = 540 - 540 * split, w = 4, h = 1080 * split, fill = ui.TEXT, opacity = 0.85 }
  end
  if lb >= 0.5 then
    local l = ui.snap(lb - 0.5)
    local left = "SDL_GPU · Vulkan"
    ui.chip(920 - (1 - l) * 40 - (text_width { left, font = ui.MONO, size = 28 } + 39), 200, left, ui.Z, ui.BG, 28, l)
    ui.chip(1000 + (1 - l) * 40, 200, "software · on the CPU", ui.PINK, ui.BG, 28, l)
  end
  local p1, p2 = T.PARITY_POPS[1], T.PARITY_POPS[2]
  if lb >= p1 then
    group({ transform = { scale = 0.6 + 0.4 * ui.snap(lb - p1), origin = { 960, 790 } } }, function()
      rect { x = 420, y = 690, w = 1080, h = 190, rx = 30, fill = ui.BG, fill_opacity = 0.88 }
      text { "GPU = software · " .. facts.PARITY, x = 960, y = 830, align = "center", font = ui.DISPLAY, size = 120,
        fill = ui.TEXT }
    end)
  end
  if lb >= p2 then
    group({ opacity = progress(lb, p2, p2 + 0.3) }, function()
      text { "gpu_check cases match · PSNR " .. facts.PSNR, x = 960, y = 940, align = "center", font = ui.SANS,
        size = 38, fill = ui.TEXT }
      text { facts.GPUS .. " · o3d v0.1.0", x = 960, y = 990, align = "center", font = ui.MONO, size = 22, fill = ui.DIM }
    end)
  end
  ui.label(lb, 6, "two renderers, one picture")
end }
