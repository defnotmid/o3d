-- The parity scene's 3D, shared by the video (drawn on the GPU) and by
-- software.lua (the same view drawn on the CPU, `--software`, into
-- media/parity_software.mp4): Oxa, glass and metal turning in the studio.

local stage = require("stage")

return function(f)
  -- a slow swing across the front: Oxa stays three-quarters on, both halves showing it
  local a = -0.5 + f.seconds * 0.14
  view3d {
    camera = { fov = 30, position = { 9.5 * math.sin(a), 2.8, 9.5 * math.cos(a) }, target = { 0, 1.35, 0 } },
    environment = stage.STUDIO, environment_intensity = 1.4,
    tone_mapping = "agx",
    lights = stage.lights(2.4),
    meshes = {
      stage.floor("#17131f", 0.6),
      stage.oxa_mesh {},
      { geometry = { type = "SphereGeometry", radius = 0.55, width_segments = 64, height_segments = 32 },
        position = { 1.6, 0.55, 1.2 }, material = { type = "MeshPhysicalMaterial", roughness = 0.04, transmission = 1,
          thickness = 1, ior = 1.5 }, cast_shadow = true },
      { geometry = { type = "TorusGeometry", radius = 0.42, tube = 0.16, radial_segments = 32, tubular_segments = 96 },
        position = { -1.7, 0.6, 1.0 }, rotation = { 0.4, f.seconds, 0 },
        material = { color = "#ffd3df", metalness = 1, roughness = 0.2 }, cast_shadow = true },
    },
  }
end
