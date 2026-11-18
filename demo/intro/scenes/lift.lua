-- The band comes in and Oxa lifts off the page: the same pixels, now
-- voxels (one RoundedBoxGeometry drawn as 636 instances, each its pixel's
-- colour), seen first exactly as the 2D was (16 px a voxel), growing their
-- depth from the middle out while the camera swings round to show them
-- solid, shadowed on a floor. Then the wordmark.

local score = require("track")
local ui = require("ui")
local stage = require("stage")

-- The camera that sees Oxa as the title drew it: fov 30, 16 px a 0.1 unit
-- voxel at 1080 px, so 6.75 units of height, 12.6 units away.
local FLAT = 6.75 / (2 * math.tan(math.rad(15)))

scene { "Lift", start = score.time("lift"), cue = "lift", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  local swing = ease(ui.SWING, lb - 0.5, 3)
  local a = lerp(0, 0.62, swing)
  local radius = lerp(FLAT, 11, swing)
  local target = { lerp(0, 2.4, swing), 1.6, 0 }
  local camera = { fov = 30, target = target,
    position = vec3_add(target, { radius * math.sin(a), lerp(0, 1.4, swing), radius * math.cos(a) }) }
  local voxels = stage.voxels { each = function(i, p, inst)
    local d = math.sqrt((p[1] - 15.5) ^ 2 + (p[2] - 15.5) ^ 2) / 22 -- 0 in the middle, ~1 at the corners
    local s = ui.soft(lb - d * 1.2)
    inst.scale = { 1, 1, math.max(0.08, s) }
    inst.position[3] = (1 - clamp01(s)) * 0.3 * hash(i, 3)
  end }
  view3d {
    camera = camera,
    environment = stage.STUDIO, environment_intensity = 0.8,
    tone_mapping = "agx",
    lights = stage.lights(),
    meshes = {
      { geometry = stage.CUBE, material = { color = "#ffffff", roughness = 0.45 }, instances = voxels,
        cast_shadow = true, receive_shadow = true },
      stage.floor(),
    },
  }
  -- the wordmark, on the third beat's bar
  local w = ui.snap(lb - 4)
  if lb >= 4 then
    group({ opacity = progress(lb, 4, 4.3), transform = { translate = { (1 - w) * 60, 0 } } }, function()
      text { "O3D", x = 1060, y = 560, font = ui.DISPLAY, size = 300, fill = ui.PINK }
    end)
  end
  if lb >= 5 then
    group({ opacity = progress(lb, 5, 5.3), transform = translate(0, (1 - ui.snap(lb - 5)) * 30) }, function()
      text { "3D for Odin, in three.js's vocabulary", x = 1066, y = 640, font = ui.SANS, size = 44, fill = ui.TEXT }
    end)
  end
  if lb >= 6 then
    ui.chip(1066, 690, "SDL_GPU (Vulkan · Metal) + a software renderer", ui.LINE, ui.PINK, 22, ui.snap(lb - 6))
  end
end }
