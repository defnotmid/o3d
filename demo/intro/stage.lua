-- The 3D the scenes share: Oxa as voxels, the studio that lights them, and
-- the floor.

local oxa = require("oxa")

local M = {}

-- A voxel's edge, in units: Oxa is 3.2 units tall.
M.VOXEL = 0.1

M.CUBE = { type = "RoundedBoxGeometry", width = M.VOXEL, height = M.VOXEL, depth = M.VOXEL, radius = 0.016, segments = 2 }

local PIXELS = oxa.pixels(oxa.DEFAULT)
local BLINK = oxa.pixels(oxa.BLINK)
M.PIXELS = PIXELS

-- Oxa's gills (the axes): their pixels glow in the Light scene.
local GILL = { ["#ff5a4e"] = true, ["#4ee07a"] = true, ["#4e9bff"] = true }

-- Oxa's voxels as instances, its feet on y = 0 and centred on x and z.
-- `opts.blink` closes its eyes; `opts.each(i, p, inst)` changes an
-- instance (`p` its pixel: col, row, colour), false drops it;
-- `opts.glow` makes the gills emissive.
function M.voxels(opts)
  opts = opts or {}
  local list, out = opts.blink and BLINK or PIXELS, {}
  local v = M.VOXEL
  for i, p in ipairs(list) do
    local inst = { position = { (p[1] - 15.5) * v, (31.5 - p[2]) * v, 0 }, color = p[3] }
    if opts.glow and GILL[p[3]] then inst.emissive = p[3] end
    if not opts.each or opts.each(i, p, inst) ~= false then out[#out + 1] = inst end
  end
  return out
end

-- Oxa as one mesh of voxels.
function M.oxa_mesh(opts)
  opts = opts or {}
  return { geometry = M.CUBE, material = opts.material or { color = "#ffffff", roughness = 0.45 },
    instances = M.voxels(opts), position = opts.position, rotation = opts.rotation, scale = opts.scale,
    cast_shadow = true, receive_shadow = true }
end

-- A studio drawn once (three.js's RoomEnvironment, as onimite's 3d.md
-- draws one): dim walls, two softboxes and a warm strip.
local STUDIO = [[
uniform float3 iResolution;
half4 main(float2 c) {
  float2 uv = c / iResolution.xy;
  float lon = uv.x * 6.2831853, lat = (0.5 - uv.y) * 3.14159265;
  float3 d = float3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  float3 wall = float3(0.22, 0.2, 0.25) + float3(0.1, 0.08, 0.1) * d.y;
  float box = smoothstep(0.88, 0.9, dot(d, normalize(float3(0.6, 0.55, 0.6))))
            + smoothstep(0.9, 0.92, dot(d, normalize(float3(-0.7, 0.4, 0.35))))
            + 0.6 * smoothstep(0.9, 0.93, dot(d, normalize(float3(0.0, 0.3, -1.0))));
  float strip = smoothstep(0.985, 0.99, 1.0 - abs(d.y - 0.15)) * step(0.0, -d.z);
  float3 c3 = wall + float3(1.0) * box + float3(1.0, 0.75, 0.8) * strip * 0.6;
  return half4(half3(clamp(c3, 0.0, 1.0)), 1);
}]]
M.STUDIO = bake { name = "studio", w = 1024, h = 512, draw = function()
  shader { sksl = STUDIO, x = 0, y = 0, w = 1024, h = 512 }
end }

-- The floor: a large dark plane at y = 0.
function M.floor(color, roughness)
  return { geometry = { type = "PlaneGeometry" }, scale = 60, rotation = { -math.pi / 2, 0, 0 },
    material = { color = color or "#1a1622", roughness = roughness or 0.85 }, receive_shadow = true }
end

-- Key and fill lights for the studio scenes.
function M.lights(key)
  return {
    { type = "HemisphereLight", color = "#fff0f4", ground_color = "#2a2033", intensity = 0.9 },
    { type = "DirectionalLight", color = "#ffffff", intensity = key or 2.6, position = { 3, 6, 4 }, cast_shadow = true,
      shadow = { map_size = 2048 } },
  }
end

return M
