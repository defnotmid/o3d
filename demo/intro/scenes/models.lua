-- glTF 2.0, three Khronos sample models (fetch.sh), one a coin: the Fox
-- (a skin and three clips), AnimatedMorphCube (morph targets) and ToyCar
-- (KHR_materials: clearcoat, sheen, transmission), on a turning stage
-- under the chorus.

local score = require("track")
local ui = require("ui")
local stage = require("stage")
local T = require("timeline")

-- Each clip of the Fox's in turn, one a bar.
local CLIPS = { "Survey", "Walk", "Run", "Run" }

scene { "Models", start = score.time("models"), cue = "models", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  local p1, p2, p3 = ui.soft(lb - T.MODEL_POPS[1]), ui.soft(lb - T.MODEL_POPS[2]), ui.soft(lb - T.MODEL_POPS[3])
  local clip = CLIPS[math.min(math.floor(lb / 4) + 1, #CLIPS)]
  local models = {}
  if p1 > 0.001 then
    models[#models + 1] = { src = "Fox.glb", position = { -3.1, 0, 0 }, rotation = { 0, 0.7, 0 }, scale = 0.024 * p1,
      animation = clip, time = f.seconds, cast_shadow = true }
  end
  if p2 > 0.001 then
    models[#models + 1] = { src = "AnimatedMorphCube.glb", position = { 0, 1.2, 0 }, rotation = { 0.3, f.seconds * 0.5, 0 },
      scale = 0.6 * p2, animation = "Square", time = f.seconds * 1.2, cast_shadow = true }
  end
  if p3 > 0.001 then
    models[#models + 1] = { src = "ToyCar.glb", position = { 3.1, 0.02, 0 }, rotation = { 0, -0.6 + f.seconds * 0.4, 0 },
      scale = 62 * p3, materials = { Fabric = { hidden = true } }, cast_shadow = true }
  end
  local a = lerp(0.12, -0.12, ease(ease_in_out, f.seconds, 16 * ui.SPB))
  view3d {
    camera = { fov = 32, position = { 10 * math.sin(a), 2.8, 10 * math.cos(a) }, target = { 0, 1.0, 0 } },
    environment = stage.STUDIO, environment_intensity = 1.4,
    tone_mapping = "agx",
    lights = stage.lights(2.4),
    meshes = { stage.floor("#17131f", 0.6) },
    models = models,
    labels = {
      ui.tag({ -3.1, 0.02, 1.2 }, "Fox.glb", "skin · " .. clip, ui.PINK, p1),
      ui.tag({ 0, 0.02, 1.2 }, "AnimatedMorphCube.glb", "morph targets", ui.Y, p2),
      ui.tag({ 3.1, 0.02, 1.2 }, "ToyCar.glb", "clearcoat · sheen · transmission", ui.Z, p3),
    },
  }
  ui.label(lb, 4, "glTF 2.0")
  text { "Khronos glTF-Sample-Assets · Fox: CC0 PixelMannen, rig CC-BY 4.0 tomkranis · AnimatedMorphCube, ToyCar: CC0",
    x = 160, y = 1004, font = ui.MONO, size = 18, fill = ui.DIM }
end }
