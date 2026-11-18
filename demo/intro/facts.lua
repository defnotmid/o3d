-- Every number on screen, with where it comes from.

local M = {}

-- tools/gpu_check (o3d's GPU against its software renderer, every case the
-- same scene drawn by both and compared), o3d v0.1.0 with oksl v0.1.0,
-- 2026-11-17: 49 of 49 cases match on an AMD Radeon RX 9070 series
-- (Vulkan, RADV) and on an Apple M4 (Metal); the parity cases' PSNR runs
-- from 43.8 dB (motion blur) to 71.1 dB (shapes) on RADV.
M.PARITY = "49 / 49"
M.PSNR = "43.8 – 71.1 dB"
M.GPUS = "Radeon (Vulkan, RADV) · Apple M4 (Metal)"

return M
