-- Light and the lens: the verse drops to half time and the room goes dark.
-- A SpotLight circles Oxa and its shadow swings round the floor; its gills
-- (the X, Y and Z axes) glow, and the view's bloom spreads them. Then the
-- camera opens its lens (f/1.4) and racks focus from Oxa to the row of
-- lights behind it.

local score = require("track")
local ui = require("ui")
local stage = require("stage")

local LAMPS = {}
for i = 1, 9 do
  LAMPS[#LAMPS + 1] = { position = { (i - 5) * 1.3, 0.35, -6.5 - (i % 3) * 1.4 }, scale = 1,
    emissive = ({ ui.X, ui.Y, ui.Z })[i % 3 + 1] }
end

scene { "Light", start = score.time("light"), cue = "light", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = "#050408" }
  -- the spot swings across the front, so Oxa stays lit as its shadow turns
  local a = math.pi / 2 + 1.1 * math.sin(f.seconds * 0.9)
  local spot = { 3.4 * math.cos(a), 5.5, 3.4 * math.sin(a) }
  local kick = score.pulse(f, { part = "drums", note = "kick", decay = 7 })
  local camera = { fov = 30, position = { 1.4, 2.1, 8.2 }, target = { 0.2, 1.5, 0 } }
  -- the lens opens on the fifth bar... and racks back to the lamps on the seventh
  if lb >= 8 then
    local rack = ease(ui.SWING, (lb - 12) * ui.SPB, 1.4 * ui.SPB * 2)
    camera.f_stop = lerp(8, 1.4, ease(ease_out, (lb - 8) * ui.SPB, 0.8))
    camera.focus = lerp(8.3, 15, rack)
  end
  view3d {
    camera = camera,
    tone_mapping = "agx",
    fog = { type = "FogExp2", color = "#050408", density = 0.045 },
    bloom = { threshold = 1, intensity = 0.55 + 0.25 * kick, radius = 18 },
    lights = {
      { type = "AmbientLight", color = "#6a5a80", intensity = 0.12 },
      { type = "SpotLight", color = "#fff1e6", intensity = 90, position = spot, target = { 0, 1.2, 0 }, angle = 0.42,
        penumbra = 0.35, decay = 2, cast_shadow = true, shadow = { map_size = 2048 } },
    },
    meshes = {
      stage.floor("#2a2433", 0.7),
      stage.oxa_mesh { glow = true, material = { color = "#ffffff", roughness = 0.45, emissive_intensity = 3 + 2 * kick } },
      { geometry = { type = "SphereGeometry", radius = 0.22 }, material = { color = "#000000", emissive = "#ffffff",
        emissive_intensity = 6 }, instances = LAMPS },
    },
  }
  ui.label(lb, 3, "light + lens")
  if lb >= 1 then ui.chip(160, 900, "SpotLight · cast_shadow", ui.LINE, ui.TEXT, 24, ui.snap(lb - 1)) end
  if lb >= 4 then ui.chip(160, 960, "emissive · bloom", ui.LINE, ui.PINK, 24, ui.snap(lb - 4)) end
  if lb >= 8 then ui.chip(1300, 960, "f_stop 1.4 · focus", ui.LINE, ui.TEXT, 24, ui.snap(lb - 8)) end
end }
