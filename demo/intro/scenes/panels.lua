-- 2D on 3D, under the breakdown (the chips alone again): a screen whose
-- picture is a panel, 2D drawn every frame (Oxa's own SVG, its parts
-- swapped on the beat); "o3d" on the floor as a ShapeGeometry, its edges
-- by the exact area each pixel covers (ocurves); and Oxa's silhouette
-- extruded with bevels, a solid from the same outline.

local score = require("track")
local ui = require("ui")
local stage = require("stage")
local T = require("timeline")

local WORD = outline { "o3d", font = ui.SANS, size = 1.6, align = "center", baseline = "middle" }
local SILHOUETTE = outline { svg = "oxa.svg", id = "silhouette", x = -1.1, y = -1.1, w = 2.2 }

local function screen(lb)
  local beat = math.floor(lb)
  return function()
    rect { x = 0, y = 0, w = 640, h = 400, fill = "#140f1d" }
    for j = 0, 9 do rect { x = 0, y = j * 40, w = 640, h = 20, fill = "#1b1426" } end
    svg { src = "oxa.svg", x = 200, y = 40, w = 240, h = 240, overrides = {
      eyes_open = { visible = beat % 4 ~= 3 }, eyes_closed = { visible = beat % 4 == 3 },
      mouth_smile = { visible = beat % 2 == 0 }, mouth_o = { visible = beat % 2 == 1 },
      gills_a = { visible = beat % 2 == 0 }, gills_b = { visible = beat % 2 == 1 } } }
    text { "panels: 2D drawn every frame", x = 320, y = 340, align = "center", font = ui.MONO, size = 26, fill = ui.PINK }
  end
end

scene { "Panels", start = score.time("panels"), cue = "panels", render = function(f)
  local lb = ui.lb(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  local p1, p2, p3 = ui.soft(lb - T.PANEL_POPS[1]), ui.soft(lb - T.PANEL_POPS[2]), ui.soft(lb - T.PANEL_POPS[3])
  local meshes, groups = { stage.floor("#120e18", 0.8) }, {}
  -- the screen: a body and its picture just in front, scaled together
  if p1 > 0.001 then
    groups[1] = { position = { -1.9, 1.6, -0.6 }, rotation = { 0, 0.35, 0 }, scale = p1, meshes = {
      { geometry = { type = "RoundedBoxGeometry", width = 3.4, height = 2.2, depth = 0.16, radius = 0.06, segments = 3 },
        material = { color = "#2a2433", roughness = 0.4 }, cast_shadow = true },
      { geometry = { type = "PlaneGeometry" }, position = { 0, 0, 0.09 }, scale = { 3.2, 2.0, 1 },
        material = { type = "MeshBasicMaterial", map = "screen" } },
    } }
  end
  if p2 > 0.001 then
    meshes[#meshes + 1] = { geometry = { type = "ShapeGeometry", shape = WORD }, position = { 0.3, 0.01, 1.3 },
      rotation = { -math.pi / 2, 0, 0 }, scale = 0.75 * p2,
      material = { type = "MeshBasicMaterial", color = ui.PINK, transparent = true } }
  end
  if p3 > 0.001 then
    meshes[#meshes + 1] = { geometry = { type = "ExtrudeGeometry", shape = SILHOUETTE, depth = 0.25, bevel_thickness = 0.04,
      bevel_size = 0.03, bevel_offset = -0.03, crease_angle = math.pi / 4 }, position = { 2.4, 1.3, 0 },
      rotation = { 0, -0.5 + f.seconds * 0.8, 0 }, scale = p3,
      material = { type = "MeshPhysicalMaterial", color = ui.PINK, roughness = 0.35, clearcoat = 1 },
      side_material = { color = "#7a1f2e", roughness = 0.6 }, cast_shadow = true }
  end
  view3d {
    camera = { fov = 32, position = { 0.6, 2.6, 8.5 }, target = { 0.2, 1.1, 0 } },
    environment = stage.STUDIO, environment_intensity = 1.1,
    tone_mapping = "agx",
    lights = stage.lights(2.0),
    panels = { screen = { w = 640, h = 400, draw = screen(lb) } },
    meshes = meshes,
    groups = groups,
  }
  ui.label(lb, 5, "2D on 3D")
  if lb >= T.PANEL_POPS[2] then ui.chip(160, 960, "ShapeGeometry · exact coverage", ui.LINE, ui.PINK, 24, ui.snap(lb - T.PANEL_POPS[2])) end
  if lb >= T.PANEL_POPS[3] then ui.chip(1140, 960, "ExtrudeGeometry · bevels", ui.LINE, ui.TEXT, 24, ui.snap(lb - T.PANEL_POPS[3])) end
end }
