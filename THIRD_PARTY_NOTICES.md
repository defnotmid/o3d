# Third-party notices

o3d is MIT licensed ([LICENSE](LICENSE)). Every license it carries is a
separate file in [LICENSES/](LICENSES); a program that ships o3d carries
these (onimite's notices list them).

## In a program using o3d

| Component | What it does | License file |
| --- | --- | --- |
| oksl | compiles ShaderMaterials (`deps:oksl`) | `LICENSES/oksl-MIT.txt` |
| ovg's curves package | exact shapes' per-pixel coverage (`deps:ovg/curves`, and `shaders/curves.slang` compiled into the curve shaders) | `LICENSES/ovg-MIT.txt` |
| Slug (Eric Lengyel's reference shaders, Copyright 2017; dual MIT OR Apache-2.0, used under MIT; the patent is dedicated to the public domain) | listed by ovg for its curves package, which o3d's exact shapes and text use | `LICENSES/Slug-MIT.txt` |
| cgltf (`vendor:cgltf`) | glTF 2.0 / GLB parsing (`gltf.odin`) | `LICENSES/cgltf-MIT.txt` |
| stb_image (`vendor:stb/image`) | texture decoding (used under MIT) | `LICENSES/stb-MIT.txt` |
| SDL 3 (`vendor:sdl3`) | the GPU renderer's device and SDL_GPU (`o3d/sdlgpu`) | `LICENSES/SDL3-Zlib.txt` |
| three.js | o3d follows its conventions and models: MeshStandardMaterial's shading, PCF shadows, equirectangular environment lighting, GLTFLoader's material and scene rules; `shape.odin` ports its ShapePath.toShapes and ExtrudeGeometry | `LICENSES/three.js-MIT.txt` |
| earcut (mapbox/earcut, as three.js carries it) | polygon triangulation with holes, ported to Odin (`triangulate_shape` in `shape.odin`) | `LICENSES/earcut-ISC.txt` |
| Odin core and vendor packages (Copyright 2016-2025 Ginger Bill) | the runtime and core library every Odin program links, and the vendor bindings above | `LICENSES/Odin-Zlib.txt` |

The standard material's shaders are compiled offline by Slang
(`shaders/build.sh`); only its output (the committed SPIR-V and MSL) is in
o3d.
