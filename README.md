# o3d

3D for Odin in three.js's vocabulary, for rendering video frames: every
frame is described from scratch (a camera, lights, geometry with materials),
so a frame is a pure function of whatever drives it. Drawn on the GPU
through SDL_GPU (Vulkan, Metal), with a software renderer that draws the
same pixels on the CPU and is the reference the GPU is tested against.
MIT licensed ([LICENSE](LICENSE)); what it carries (oksl, cgltf,
stb_image, SDL 3, three.js's models, Odin's packages) is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Conventions (three.js's)

What is known about three.js carries over:

- Right-handed and Y up; the camera looks down -Z.
- Rotations are Euler angles in radians, applied in X, Y, Z order; a
  transform is position, rotation and scale (`compose`), like Object3D.
- The camera's field of view is vertical, in degrees (PerspectiveCamera);
  an orthographic camera takes the view's height.
- Colours are sRGB; lighting is linear; the output is sRGB with no tone
  mapping.
- Materials are MeshStandardMaterial (`color`, `metalness`, `roughness`,
  `emissive`, `emissive_intensity`, `opacity`, `side`, texture maps
  `color_map`, `normal_map`, `roughness_map` (green), `metalness_map`
  (blue; glTF packs both in one texture), `emissive_map`, and the
  booleans as a bit set: `flags = {.Transparent, .Flat_Shading}`) or
  MeshBasicMaterial (`.Basic`), or ShaderMaterial: your own vertex and
  fragment shaders in GLSL ES (three.js's GLSL 1 style with `attribute`,
  `varying` and `gl_FragColor`, or GLSL 3 with `glsl3 = true`), with
  three.js's built-in uniforms (`modelMatrix`, `modelViewMatrix`,
  `projectionMatrix`, `viewMatrix`, `normalMatrix`, `cameraPosition`,
  `isOrthographic`) and attributes (`position`, `normal`, `uv`), compiled by
  [oksl](../oksl) on both renderers. The output is written as the shader
  leaves it; `.Transparent` blends it, `side` culls as for any material.
  `projectionMatrix` maps depth to 0..1 (SDL_GPU's clip space) rather than
  WebGL's -1..1, which matters only to shaders that build their own.
- Lights are AmbientLight, HemisphereLight, DirectionalLight (from
  `position` toward `target`) and PointLight (`distance`, `decay`), in
  three.js's units: a white directional light of intensity 1 lights a white
  surface facing it to 1/pi of white, so scenes use intensities around 2 to 3.
- Shadows are DirectionalLightShadow's with PCF: a directional light with
  `cast_shadow` (the first) and its `shadow` settings (`map_size`,
  `height` of its square orthographic view, `near`, `far`, `bias`,
  `normal_bias`; zero takes three.js's 512, 10, 0.5, 500), draws with
  `.Cast_Shadow` and `.Receive_Shadow` in their material's flags (three.js
  has them on the mesh; a draw's material is its own). Casters draw their
  other side (three.js's default shadowSide) and plain geometry;
  ShaderMaterials cast but do not receive.
- Geometry takes the three.js parameters: `box(w, h, d, segments...)`,
  `sphere(radius, widthSegments, heightSegments)`, `plane`, `cylinder`
  (radiusTop 0 is a cone), `torus`. The same parameters give the same id,
  so a frame may ask for `box(1, 1, 1)` every time; `primitive(key)` makes
  one from its `Geometry_Key` alone. Geometry is kept (never freed), so a
  shape changes over time through the draw's transform, as three.js
  animates `mesh.scale`: `sphere(geos, 1)` scaled by `compose`, not
  `sphere(geos, radius(t))`, which makes a new mesh every frame it differs.
  `geometries_made` lets a host check that it stops growing.

The output is premultiplied RGBA, transparent where nothing was drawn, to
composite over anything.

## Use

```odin
geos: o3d.Geometries                       // kept: geometry is made once
textures: o3d.Textures                     // kept: textures and mipmaps are made once
albedo, _ := o3d.texture_load(&textures, "brick.png")
scene: o3d.Scene                           // filled every frame
o3d.scene_clear(&scene)
scene.camera = {position = {0, 2, 6}, target = {0, 0.5, 0}, fov = 40, near = 0.1, far = 100}
append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.2, 0.15, 0.2}, intensity = 1.5})
append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {3, 5, 2},
	cast_shadow = true, shadow = {map_size = 1024, bias = -0.0005}})
m := o3d.MATERIAL_DEFAULT
m.color, m.roughness, m.color_map = {1, 0.35, 0.3}, 0.4, albedo
m.flags = {.Cast_Shadow, .Receive_Shadow}
append(&scene.draws, o3d.Draw{o3d.box(&geos, 1, 1, 1), o3d.compose({0, 0.5, 0}, {0, t, 0}), m})

// glTF 2.0 / GLB: loaded once into flat tables, drawn per frame at time t
duck, _ := o3d.gltf_load("duck.glb", &geos, &textures)
o3d.gltf_draw(&scene, &duck, o3d.compose({-1.5, 0, 0}), anim = 0, time = t)

// ShaderMaterial: compiled once (same sources, same id), uniforms per frame
shaders: o3d.Shaders                       // kept, like geometry
wave, errors := o3d.shader(&shaders, vertex_glsl, fragment_glsl)   // errors: "Fragment:3:7: ..."
w := o3d.MATERIAL_DEFAULT
w.shader = wave
w.uniforms = o3d.scene_uniforms(&scene, {"uTime", {t}}, {"uColor", {0.3, 0.8, 0.4}})
append(&scene.draws, o3d.Draw{o3d.sphere(&geos, 1), o3d.compose(), w})

// GPU (o3d/sdlgpu): borrows the host's SDL_GPU device, or makes a headless one
g: o3d_sdlgpu.GPU
o3d_sdlgpu.init(&g, device)
out := o3d_sdlgpu.target(&g, 1920, 1080)
o3d_sdlgpu.render(&g, &scene, &geos, &shaders, out, 1920, 1080, cmd, &textures)   // or nil cmd: submit and wait

// CPU: the same pixels, rows banded on the caller's thread pool
sw: o3d.Software
pixels := o3d.software_render(&sw, &scene, &geos, &shaders, 1920, 1080, pool, &textures)
```

o3d imports oksl as `deps:oksl`, so builds pass `-collection:deps=..`
(the directory holding both): `odin test . -collection:deps=..`,
`odin run bench -collection:deps=.. -o:speed` runs the zero-allocation
software benchmark, and `odin run demo -collection:deps=.. -- out_dir`
renders a test scene both ways to PNGs.

## Design

- **Data:** the scene is flat tables (lights, draws: geometry id, model
  matrix, material) refilled each frame, the draws a structure of arrays
  (`#soa`) since the per-frame passes read a column or two; geometry is
  the only retained state: one array of interleaved position / normal / uv vertices and one
  of 32-bit indices shared by every geometry, a geometry being its ranges
  in them. The GPU mirrors the two arrays in two buffers, uploading what
  is new at the start of a frame.
- **glTF (`gltf.odin`):** `gltf_load` and `gltf_parse` (`vendor:cgltf`)
  unpack meshes into `Geometries`, images into `Textures`, metallic-roughness
  materials into `Material`, the node hierarchy into a topologically sorted
  `#soa` array (`parent < i`), and keyframe animations (`STEP`, `LINEAR`,
  `CUBICSPLINE`) into flat time/value arrays. `gltf_draw` evaluates the
  hierarchy and animation at time `t` in a single forward pass with zero
  heap allocations. As three.js's GLTFLoader: the default scene; OPAQUE
  ignores alpha, BLEND blends, MASK draws opaque (no alpha cutoff yet);
  the metallicRoughness texture is both roughness and metalness map; no
  normals means flat shading. Files are validated (`cgltf.validate`) and
  out-of-range indices rejected. Not imported: skins, morph targets,
  cameras, lights, `KHR_*` extensions, texture transforms, a second uv set.
- **Tables:** primitives are made by `PRIMITIVES`, indexed by `Primitive`.
  Each frame, `shading_lights` and `shading_materials` turn the lights and
  the draws' materials into their shading form (linear colours, the
  opacity drawn, `Shading_Flags`) once; both renderers read those, and the
  GPU's uniforms are them, packed. Boolean sets are bit sets
  (`Material_Flags`, `Shading_Flags`, a pixel's covered samples).
- **Shader contract:** `Light_Kind`'s values and `Shading_Flag`'s bits are
  the shader's `LIGHT_*` and `FLAG_*` constants; a test reads
  `shaders/o3d.slang` and checks them.
- **Shading** is one model in two places, `shade` in scene.odin and
  `shaders/o3d.slang`: Lambert diffuse plus GGX specular with Smith
  correlated visibility and Schlick Fresnel, plus equirectangular
  environment lighting (`scene.environment`, `scene.environment_intensity`)
  and background (`scene.background`, `scene.background_intensity`,
  `scene.background_blurriness`), as three.js's MeshStandardMaterial
  computes it. When a texture is used as `scene.environment` (or blurred
  `scene.background`), the texture has derived maps, as it has mipmaps: a
  GGX-prefiltered radiance mip chain (roughness 0..1) and a
  cosine-convolved diffuse irradiance map in `Textures`, sampled alike by
  both renderers. `environment_prefilter(textures, id)` makes them; it is
  heavy, so a host calls it at load (a render that meets a texture without
  them makes them then, in frame time); `texture_environment` looks them up.
  Changing one means changing the other; the tests compare the two
  renderers.
- **Anti-aliasing:** 4x MSAA on the GPU; the software renderer keeps four
  samples per pixel at the same positions, shades once per pixel per
  triangle, and resolves the same way.
- **Software phases,** each spread over the caller's pool: chunks of the
  drawing order (up to 8192 triangles, so one large mesh splits too) are
  transformed, clipped and counted per band of 32 rows; after a prefix sum
  each chunk lists its triangles in the bands, so a band's list keeps the
  drawing order; then each band clears its rows, draws its list and
  resolves its pixels. A frame is the same bytes with or without a pool
  (tested). Triangles are a structure of arrays, as counting and listing
  read only their row spans. Vertices stay interleaved: the GPU reads them
  as one buffer and the rasterizer gathers whole vertices by index.
- **Shadows** (shadow.odin) are drawn first: the casters' depth from the
  light (`shadow_casters`, culled to its view), one sample per texel
  centre, nearest kept; the software renderer runs the same phases in a
  depth-only mode at the map's size, the GPU renders into an R32_FLOAT
  colour target (sampleable on every backend) with a depth buffer. A
  receiver's pixel scales the shadow light's direct light by
  `shadow_factor`: a 3x3 grid of bilinear depth comparisons (a 4x4 block
  of texels weighed 1 - f, 1, 1, f per axis), edges repeated, outside the
  light's view lit. The shader has the same arithmetic (`shadow_factor` in
  `shaders/o3d.slang`); `FLAG_RECEIVE_SHADOW` joins the shader contract.
- **Order:** draws whose bounding sphere (the geometry's radius, scaled
  with the draw) is outside the view, or whose shader failed, are skipped;
  then opaque draws as listed, then blended ones (transparent with opacity
  under 1, or a transparent ShaderMaterial) farthest first, without writing
  depth (`draw_order`, shared by both renderers).
- **Shaders:** the standard material's are Slang compiled offline
  (`shaders/build.sh`) to committed SPIR-V and MSL, as ovg's are.
  ShaderMaterials are compiled at run time by oksl, which is plain Odin in
  the binary (no external toolchain), to SPIR-V and MSL and to programs
  its CPU interpreter runs. Their table (`Shaders`) keeps each compiled
  pair by id with what it reads, found once: the built-ins' block offsets
  per stage, the attributes' locations, the varyings' floats and the flat
  ones as a list. `BUILTINS` and `VERTEX_ATTRIBUTES` make three.js's
  prelude for each stage and say what to fill; `shader_blocks` writes every
  custom draw's uniform bytes once a frame, which the GPU pushes and the
  software renderer runs on. A shader that fails is kept with its errors
  and its draws left out by both renderers.
- **Software ShaderMaterials:** a chunk runs its draw's vertex stage once
  over the vertices it uses (the standard material's too: world position
  and normal), and a triangle's corners carry those floats, divided by w
  except flat ones. A band shades a run of one custom draw's triangles as
  2x2 quads (helper lanes for derivatives) in one oksl call, then writes
  them through the same coverage, depth and blending as the standard
  material, in triangle order.
- Errors are values (`Error`); no threads of its own: the software
  renderer runs its phases on the caller's `thread.Pool`.

## Next

v0.0.1 is the first tag, for onimite's integration (the repository is
private for now). In order, with the size of each next to shadows (one
sizeable change):

1. onimite (medium to large, mostly in onimite): a `view3d{}` node drawing
   into its own texture, composited like any layer; a group's 2D drawing as
   a texture on a surface; the Lua API in three.js's names; inspect
   checks; the docs page mapping three.js terms.
2. ShaderMaterial textures (small here): `uniform sampler2D` bound to
   `Textures`; `texelFetch` / `textureSize` and CPU implicit-LOD
   `texture()` wait on oksl (deferred there).

Before onimite depends on o3d: run the tests on macOS. The Metal path
(the committed MSL, the R32_FLOAT shadow map, oksl's MSL for
ShaderMaterials) is generated but has only been run on Vulkan.
