-- three.js's materials, one a coin: metal, a clear coat, glass refracting
-- a small Oxa behind it, velvet sheen, and toon bands with an outline, in a
-- studio drawn once (bake{}) that lights and reflects them.

local score = require("track")
local ui = require("ui")
local stage = require("stage")
local T = require("timeline")

local ITEMS = {
  { title = "MeshStandardMaterial", detail = "metalness 1 · roughness 0.15", colour = ui.PINK,
    geometry = { type = "SphereGeometry", width_segments = 64, height_segments = 32 }, size = 0.62,
    material = { type = "MeshStandardMaterial", color = "#ffd3df", metalness = 1, roughness = 0.15 } },
  { title = "clearcoat", detail = "MeshPhysicalMaterial", colour = ui.X,
    geometry = { type = "TorusGeometry", radius = 0.45, tube = 0.2, radial_segments = 32, tubular_segments = 96 },
    material = { type = "MeshPhysicalMaterial", color = "#e8352b", metalness = 0.4, roughness = 0.55, clearcoat = 1,
      clearcoat_roughness = 0.04 } },
  { title = "transmission", detail = "ior 1.5 · thickness 1.2", colour = "#ffffff",
    geometry = { type = "SphereGeometry", width_segments = 64, height_segments = 32 }, size = 0.68,
    material = { type = "MeshPhysicalMaterial", color = "#ffffff", roughness = 0.04, transmission = 1, thickness = 1.2,
      ior = 1.5 } },
  { title = "sheen", detail = "velvet: sheen_color", colour = ui.ROSE,
    geometry = { type = "RoundedBoxGeometry", width = 1.05, height = 0.75, depth = 1.05, radius = 0.3, segments = 6 },
    material = { type = "MeshPhysicalMaterial", color = "#3b1a4a", roughness = 0.8, sheen = 1, sheen_color = "#ff9ec0",
      sheen_roughness = 0.35 } },
  { title = "MeshToonMaterial", detail = "outline_width 0.025", colour = ui.Y,
    geometry = { type = "LatheGeometry", segments = 48,
      points = { { 0.01, -0.6 }, { 0.42, -0.58 }, { 0.5, -0.3 }, { 0.36, 0.05 }, { 0.22, 0.32 }, { 0.3, 0.55 }, { 0.36, 0.62 } } },
    material = { type = "MeshToonMaterial", color = "#4ee07a", outline_width = 0.025, outline_color = "#1b1220" } },
}

local SPACING = 2.0

scene { "Materials", start = score.time("materials"), cue = "materials", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  local meshes, labels = { stage.floor("#17131f", 0.6) }, {}
  for i, it in ipairs(ITEMS) do
    local pop = ui.soft(lb - T.MATERIAL_POPS[i])
    local x = (i - 3) * SPACING
    -- a pedestal, and the piece turning on it
    meshes[#meshes + 1] = { geometry = { type = "CylinderGeometry", radius_top = 0.55, radius_bottom = 0.6, height = 0.5,
      radial_segments = 48 }, position = { x, 0.25, 0 }, material = { color = "#2a2433", roughness = 0.5 },
      cast_shadow = true, receive_shadow = true }
    if pop > 0.001 then
      meshes[#meshes + 1] = { geometry = it.geometry, material = it.material, position = { x, 1.25, 0 },
        rotation = { i == 2 and 0.9 or 0, f.seconds * 0.6 + i, 0 }, scale = math.max(pop, 0.001) * (it.size or 1), cast_shadow = true,
        receive_shadow = true }
    end
    labels[#labels + 1] = ui.tag({ x, 0.02, 0.9 }, it.title, it.detail, it.colour, pop)
  end
  -- the small Oxa behind the glass
  meshes[#meshes + 1] = stage.oxa_mesh { position = { 0, 0.5, -1.4 }, scale = 0.42 }
  local a = lerp(-0.18, 0.18, ease(ease_in_out, f.seconds, 16 * ui.SPB))
  view3d {
    camera = { fov = 34, position = { 9.6 * math.sin(a), 3.0, 9.6 * math.cos(a) }, target = { 0, 0.9, 0 } },
    environment = stage.STUDIO, environment_intensity = 2.6,
    tone_mapping = "agx",
    lights = stage.lights(2.2),
    meshes = meshes,
    labels = labels,
  }
  ui.label(lb, 2, "materials")
end }
