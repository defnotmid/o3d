-- Where things happen, in beats from the start of the video (4 beats a
-- bar), read from the song (track.lua): its sections are the scenes, its
-- coin blips the pops the picture answers to, so picture and sound share
-- one list.

local score = require("track")

local T = {}

-- The beat at a position in the song's grammar ("lift", "outro@1m").
local function at(pos) return score.beat(score.time(pos)) end
-- The song's hits of `part` in section `section`, in beats into it.
local function hits(part, section)
  local out, start = {}, at(section)
  for i, t in ipairs(score.hits { part = part, range = section }) do out[i] = score.beat(t) - start end
  return out
end

T.TITLE, T.LIFT, T.MATERIALS, T.LIGHT, T.MODELS, T.PANELS, T.PARITY, T.OUTRO = at("title"), at("lift"),
  at("materials"), at("light"), at("models"), at("panels"), at("parity"), at("outro")
T.END = at("tail")
T.TOTAL = score.beat(score.duration)

T.MATERIAL_POPS = hits("blips", "materials") -- the five materials, one a pop
T.MODEL_POPS = hits("blips", "models")       -- the three models
T.PANEL_POPS = hits("blips", "panels")       -- the panel, the shape text, the extrusion
T.PARITY_POPS = hits("blips", "parity")      -- the parity's two numbers

return T
