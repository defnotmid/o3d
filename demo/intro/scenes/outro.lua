-- The last chorus: the wordmark extruded in 3D beside Oxa's voxels, over
-- the soundtrack's spectrum; the song lands on its tonic (Dmaj7) only
-- here, and Oxa blinks.

local score = require("track")
local ui = require("ui")
local stage = require("stage")

local WORD = outline { "O3D", font = ui.DISPLAY, size = 2.6, align = "center", baseline = "middle" }

scene { "Outro", start = score.time("outro"), cue = "outro", render = function(f)
  local lb = ui.lb(f)
  local fade = 1 - progress(lb, 13, 15.5)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  local bands = spectrum_centered(ui.spectrum(f, 48))
  for i, v in ipairs(bands) do
    local h = v * 150 + 6
    rect { x = 172 + (i - 1) * 32.5, y = 1040 - h, w = 22, h = h, rx = 6, opacity = fade * 0.8,
      fill = linear_gradient { x1 = 0, y1 = 0, x2 = 0, y2 = 1, units = "bbox", stops = { { 0, ui.PINK }, { 1, ui.Z } } } }
  end
  local s = ui.soft(lb)
  local kick = score.pulse(f, { part = "drums", note = "kick", decay = 8 })
  group({ opacity = fade }, function()
    view3d {
      camera = { fov = 30, position = { 0.8, 2.2, 10.5 }, target = { 0.6, 1.6, 0 } },
      environment = stage.STUDIO, environment_intensity = 1.4,
      tone_mapping = "agx",
      lights = stage.lights(2.4),
      meshes = {
        { geometry = { type = "ExtrudeGeometry", shape = WORD, depth = 0.5, bevel_thickness = 0.06, bevel_size = 0.04,
          bevel_offset = -0.04, crease_angle = math.pi / 4 }, position = { 2.1, 1.75, -0.25 },
          rotation = { 0, 0.25 * math.sin(f.seconds * 0.8), 0 }, scale = math.max(s, 0.001) * (1 + 0.015 * kick),
          material = { type = "MeshPhysicalMaterial", color = ui.PINK, roughness = 0.3, clearcoat = 1 },
          side_material = { color = "#7a1f2e", roughness = 0.5 }, cast_shadow = true },
        stage.oxa_mesh { position = { -2.4, 0, 0 }, rotation = { 0, 0.35, 0 }, scale = 0.85,
          blink = lb > 10 and lb < 10.4 },
      },
    }
    group({ opacity = progress(lb, 2, 2.4) }, function()
      text { "3D for Odin, in three.js's vocabulary · github.com/defnotmid/o3d", x = 960, y = 760, align = "center",
        font = ui.SANS, size = 36, fill = ui.TEXT }
    end)
    group({ opacity = progress(lb, 3, 3.4) }, function()
      text { "this video: an onimite script drawn by o3d · o3d/demo/intro", x = 960, y = 806, align = "center",
        font = ui.MONO, size = 22, fill = ui.DIM }
    end)
  end)
end }
