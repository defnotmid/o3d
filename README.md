# o3d

3D for Odin in three.js's vocabulary, for rendering video frames: every
frame is described from scratch (a camera, lights, geometry with materials),
so a frame is a pure function of whatever drives it. Drawn on the GPU
through SDL_GPU (Vulkan, Metal), with a software renderer that draws the
same pixels on the CPU and is the reference the GPU is tested against.
MIT licensed ([LICENSE](LICENSE)).

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
  `emissive`, `emissive_intensity`, `opacity`, `transparent`, `side`,
  `flat_shading`) or MeshBasicMaterial (`basic`).
- Lights are AmbientLight, HemisphereLight, DirectionalLight (from
  `position` toward `target`) and PointLight (`distance`, `decay`), in
  three.js's units: a white directional light of intensity 1 lights a white
  surface facing it to 1/pi of white, so scenes use intensities around 2 to 3.
- Geometry takes the three.js parameters: `box(w, h, d, segments...)`,
  `sphere(radius, widthSegments, heightSegments)`, `plane`, `cylinder`
  (radiusTop 0 is a cone), `torus`. The same parameters give the same id,
  so a frame may ask for `box(1, 1, 1)` every time.

The output is premultiplied RGBA, transparent where nothing was drawn, to
composite over anything.

## Use

```odin
geos: o3d.Geometries                       // kept: geometry is made once
scene: o3d.Scene                           // filled every frame
o3d.scene_clear(&scene)
scene.camera = {position = {0, 2, 6}, target = {0, 0.5, 0}, fov = 40, near = 0.1, far = 100}
append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.2, 0.15, 0.2}, intensity = 1.5})
append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {3, 5, 2}})
m := o3d.MATERIAL_DEFAULT
m.color, m.roughness = {1, 0.35, 0.3}, 0.4
append(&scene.draws, o3d.Draw{o3d.box(&geos, 1, 1, 1), o3d.compose({0, 0.5, 0}, {0, t, 0}), m})

// GPU (o3d/sdlgpu): borrows the host's SDL_GPU device, or makes a headless one
g: o3d_sdlgpu.GPU
o3d_sdlgpu.init(&g, device)
out := o3d_sdlgpu.target(&g, 1920, 1080)
o3d_sdlgpu.render(&g, &scene, &geos, out, 1920, 1080, cmd)   // or nil cmd: submit and wait

// CPU: the same pixels, rows banded on the caller's thread pool
sw: o3d.Software
pixels := o3d.software_render(&sw, &scene, &geos, 1920, 1080, pool)
```

`odin run demo -collection:deps=.. -- out_dir` renders a test scene both
ways to PNGs.

## Design

- **Data:** the scene is flat tables (lights, draws: geometry id, model
  matrix, material) refilled each frame; geometry is the only retained
  state, interleaved position / normal / uv vertices with 32-bit indices,
  uploaded to the GPU on first use.
- **Shading** is one model in two places, `shade` in scene.odin and
  `shaders/o3d.slang`: Lambert diffuse plus GGX specular with Smith
  correlated visibility and Schlick Fresnel, as three.js's
  MeshStandardMaterial computes it without an environment map. Changing one
  means changing the other; the tests compare the two renderers.
- **Anti-aliasing:** 4x MSAA on the GPU; the software renderer keeps four
  samples per pixel at the same positions, shades once per pixel per
  triangle, and resolves the same way.
- **Order:** opaque draws as listed, then blended ones (transparent with
  opacity under 1) farthest first, without writing depth (`draw_order`).
- **Shaders** are Slang compiled offline (`shaders/build.sh`) to committed
  SPIR-V and MSL, as ovg's are; o3d needs no shader compiler at run time.
- Errors are values (`Error`); no threads of its own: the software
  renderer bands rows on the caller's `thread.Pool`.

## Next

1. Shadows: a shadow map for one directional light (both renderers).
2. Environment lighting: an equirectangular image as image-based light and
   background (metals reflect something), as three.js's `scene.environment`.
3. Textures: `map`, `normalMap`, `roughnessMap`, `emissiveMap` with uvs.
4. glTF import: meshes, materials, node transforms, baked animation sampled
   at a time.
5. onimite: a `view3d{}` node drawing into its own texture, composited like
   any layer; a group's 2D drawing as a texture on a surface; the Lua API in
   three.js's names; inspect checks; the docs page mapping three.js terms.
6. Custom materials in three.js's `ShaderMaterial` form (GLSL ES 3.00
   vertex and fragment shaders with its built-in uniforms and attributes),
   compiled by oksl once it has a vertex stage.
