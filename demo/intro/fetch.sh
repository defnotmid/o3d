#!/bin/sh
# Downloads the glTF models the Models scene draws, from Khronos'
# glTF-Sample-Assets at a pinned commit, and checks each against its
# SHA-256. They aren't in the repository: run this once before rendering
# (media/.gitignore keeps them out of commits). Licences are in README.md.
set -eu
cd "$(dirname "$0")/media"
BASE=https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/f0920f5b32d172a30efe3f9257e965918831e6ea/Models
sum() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
fetch() { # name sha256 credit
	if [ -f "$1.glb" ] && [ "$(sum "$1.glb")" = "$2" ]; then return; fi
	curl -fsSL -o "$1.glb" "$BASE/$1/glTF-Binary/$1.glb"
	if [ "$(sum "$1.glb")" != "$2" ]; then echo "fetch.sh: $1.glb isn't the file pinned here" >&2; rm -f "$1.glb"; exit 1; fi
	echo "fetched $1.glb: $3"
}
fetch Fox d97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7 "model CC0 (PixelMannen), rig and animations CC-BY 4.0 (tomkranis; AsoboStudio and scurest)"
fetch AnimatedMorphCube 214ee56160a50dbf22543a1d66dbf860986e87f0efac3d89feac1359d0e6aeab "CC0 (Microsoft)"
fetch ToyCar 01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c "CC0 (Guido Odendahl, Eric Chadwick)"
