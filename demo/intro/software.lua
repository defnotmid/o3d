-- The parity scene's 3D alone, as long as the scene, for o3d's software
-- renderer to draw: the right half of the video's Parity scene plays it.
--
--     onimite render demo/intro/software.lua --software -o demo/intro/media/parity_software.mp4 --qp 12

local score = require("track")
local ui = require("ui")
local view = require("parity_view")

video { width = 1920, height = 1080, fps = 60, background = ui.BG }

scene { "Parity", duration = { frames = score.frame("outro", 60) - score.frame("parity", 60) }, render = function(f)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  view(f)
end }
