#!/usr/bin/env bash
# Compiles o3d.slang, background.slang, depth.slang and post.slang to the SPIR-V (Vulkan) and MSL
# (Metal) blobs the SDL_GPU backend #loads. Only needed after editing a
# shader; the outputs are committed. DXIL (D3D12) is produced when slangc's
# dxc works; Windows otherwise runs on Vulkan.
set -euo pipefail
cd "$(dirname "$0")"

# ovg's curves.slang (exact shapes' coverage) from its checkout beside o3d.
FLAGS=(-matrix-layout-column-major -O2 -I . -I ../../ovg/shaders)
# The layout of the curves words that curves.slang reads (ovg's
# curves.FORMAT), from the same checkout: o3d_sdlgpu refuses to build
# against an ovg with another.
FORMAT=$(sed -n 's/^FORMAT :: \([0-9][0-9]*\).*/\1/p' ../../ovg/curves/curves.odin)
if [ -z "$FORMAT" ]; then
	echo "no FORMAT in ../../ovg/curves/curves.odin" >&2
	exit 1
fi
UNITS=(
	o3d:vert:vs_main
	o3d:skinned_vert:vs_skinned
	o3d:curves_vert:vs_curves
	o3d:frag:fs_main
	o3d:shadow:fs_shadow
	o3d:curves_frag:fs_curves
	o3d:sprite_frag:fs_sprite
	o3d:shadow_curves:fs_shadow_curves
	background:vert:vs_main
	background:frag:fs_main
	depth:vert:vs_main
	depth:frag:fs_main
	post:vert:vs_main
	post:output:fs_output
	post:focus_tiles:fs_focus_tiles
	post:focus_near:fs_focus_near
	post:focus_half:fs_focus_half
	post:focus:fs_focus
)
for unit in "${UNITS[@]}"; do
	IFS=: read -r mod suffix entry <<<"$unit"
	slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target spirv -o "$mod.$suffix.spv"
	slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target metal -o "$mod.$suffix.msl"
	if ! slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target dxil -o "$mod.$suffix.dxil" 2>/dev/null; then
		rm -f "$mod.$suffix.dxil"
		echo "  note: no DXIL for $mod.$suffix (slangc's dxc failed)" >&2
	fi
	spirv-val --target-env vulkan1.2 "$mod.$suffix.spv"
	echo "  $mod.slang:$entry -> $mod.$suffix.{spv,msl}"
done
cat >../sdlgpu/curves_format.odin <<EOF
package o3d_sdlgpu

// Written by shaders/build.sh: the ovg curves format (curves.FORMAT) the
// committed curve shaders read.
SHADERS_CURVES_FORMAT :: $FORMAT
EOF
echo "  curves format $FORMAT -> sdlgpu/curves_format.odin"

