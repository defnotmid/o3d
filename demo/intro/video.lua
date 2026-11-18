-- o3d in 48 seconds: an onimite script (Odin + Lua video framework,
-- https://github.com/defnotmid/onimite), drawn by o3d, the 3D it shows.
-- Music included: the soundtrack is an odawg song (track.lua), chiptune
-- over a live band at 140 BPM, and every scene starts on its section, every
-- pop on a coin blip, every flash on an impact (timeline.lua).
-- Numbers on screen come from facts.lua, each with its source.
--
--     sh demo/intro/fetch.sh                                   # once: the glTF models
--     onimite render demo/intro/software.lua --software -o demo/intro/media/parity_software.mp4 --qp 12
--     onimite render demo/intro/video.lua -o demo/intro/demo.mp4 --qp 22

local score = require("track")
local ui = require("ui")
local T = require("timeline")

local SECTIONS = { "01 / TITLE", "01 / LIFT", "02 / MATERIALS", "03 / LIGHT + LENS", "04 / GLTF 2.0", "05 / 2D ON 3D",
  "06 / PARITY", "07 / OUTRO" }
local STARTS = { T.TITLE, T.LIFT, T.MATERIALS, T.LIGHT, T.MODELS, T.PANELS, T.PARITY, T.OUTRO }

cue("impact", score.hits { part = "fx" }, { hit = "score" })

local function hud(f)
  local b = score.beat(f)
  if b >= T.END then return end
  local section = 1
  for i, s in ipairs(STARTS) do if b >= s then section = i end end
  group({ opacity = 0.8 }, function()
    text { "O3D · 3D FOR ODIN", x = 160, y = 66, font = ui.MONO, size = 18, fill = ui.DIM }
    text { string.format("140 BPM   BAR %02d.%d", math.floor(b / 4) + 1, math.floor(b % 4) + 1), x = 1730, y = 66,
      align = "right", font = ui.MONO, size = 18, fill = ui.DIM }
    rect { x = 1744, y = 52, w = 16, h = 16, fill = ui.PINK, opacity = 0.2 + 0.8 * pulse(b, 7) }
    text { SECTIONS[section], x = 1760, y = 1050, align = "right", font = ui.MONO, size = 18, fill = ui.DIM }
    rect { x = 160, y = 1062, w = 1600, h = 1, fill = ui.LINE }
    rect { x = 160, y = 1061, w = math.max(1600 * b / T.TOTAL, 1), h = 3, fill = ui.DIM }
  end)
end

video {
  width = 1920, height = 1080, fps = 60, background = ui.BG,
  duration = score.duration,
  audio = { { "score", bus = "music", fade_out = 0.5 } },
  buses = { music = {} },
  -- lifts the song to about -14 LUFS under the limiter's -2 dBTP (README.md)
  master_gain_db = 4.5,
  render = function(f)
    local kick = score.pulse(f, { part = "drums", note = "kick", decay = 9 })
    group({ transform = { scale = 1 + kick * 0.006, origin = { 960, 540 } } }, function() render_scenes(f) end)
    local flash = score.pulse(f, { part = "fx", decay = 5 })
    if flash > 0.01 then rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.TEXT, opacity = flash * 0.45 } end
    hud(f)
  end,
}

require("scenes.title")
require("scenes.lift")
require("scenes.materials")
require("scenes.light")
require("scenes.models")
require("scenes.panels")
require("scenes.parity")
require("scenes.outro")
