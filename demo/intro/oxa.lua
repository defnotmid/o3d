-- Oxa, o3d's mascot, as pixels: o3d's assets/logo.svg (media/oxa.svg is a
-- link to it) read at load, each layer's paths filled at the 32 x 32 pixel
-- centres. The file draws every colour as one path of axis-aligned
-- rectangles (M, H, V and Z only), so a row's fill is the spans between
-- its vertical edges, even-odd. The layers are its groups by id: gills_a
-- and gills_b (the axes' two poses), body, eyes_open and eyes_closed,
-- mouth_smile and mouth_o, and the hidden silhouette.

local M = {}

M.SIZE = 32

-- A path's vertical edges: { x, y0, y1 } with y0 < y1.
local function edges(d)
  local out, x, y, sx, sy = {}, 0, 0, 0, 0
  for cmd, args in d:gmatch("([MHVZ])([^MHVZ]*)") do
    local n = {}
    for v in args:gmatch("[-%d.]+") do n[#n + 1] = tonumber(v) end
    if cmd == "M" then
      x, y, sx, sy = n[1], n[2], n[1], n[2]
    elseif cmd == "H" then
      x = n[1]
    elseif cmd == "V" then
      out[#out + 1] = { x, math.min(y, n[1]), math.max(y, n[1]) }
      y = n[1]
    else -- Z: back to the subpath's start, vertical when it closes one
      if x == sx and y ~= sy then out[#out + 1] = { x, math.min(y, sy), math.max(y, sy) } end
      x, y = sx, sy
    end
  end
  return out
end

-- The pixels a path fills: grid[row][col] = colour, rows from the top.
local function fill(grid, d, colour)
  local e = edges(d)
  for row = 0, M.SIZE - 1 do
    local yc, xs = row + 0.5, {}
    for _, v in ipairs(e) do
      if v[2] < yc and yc < v[3] then xs[#xs + 1] = v[1] end
    end
    table.sort(xs)
    for i = 1, #xs - 1, 2 do
      for col = math.floor(xs[i] + 0.5), math.floor(xs[i + 1] - 0.5) do
        grid[row] = grid[row] or {}
        grid[row][col] = colour
      end
    end
  end
end

-- { name = { { col, row, colour }, ... } } for every layer, in its paths'
-- order (a later path over an earlier one).
local function read()
  local f = assert(io.open(SCRIPT_DIR .. "/media/oxa.svg", "r"))
  local svg = f:read("a")
  f:close()
  local layers = {}
  for id, body in svg:gmatch('<g id="([%w_]+)"[^>]*>(.-)</g>') do
    local grid = {}
    for colour, d in body:gmatch('<path fill="(#%x+)" d="([^"]+)"') do fill(grid, d, colour) end
    local px = {}
    for row = 0, M.SIZE - 1 do
      for col = 0, M.SIZE - 1 do
        local c = grid[row] and grid[row][col]
        if c then px[#px + 1] = { col, row, c } end
      end
    end
    layers[id] = px
  end
  return layers
end

M.layers = shared("oxa", read)

-- The pixels of `names` (layers), later layers over earlier ones: a list
-- of { col, row, colour }.
function M.pixels(names)
  local grid, out = {}, {}
  for _, name in ipairs(names) do
    for _, p in ipairs(M.layers[name]) do grid[p[2] * M.SIZE + p[1]] = p[3] end
  end
  for k = 0, M.SIZE * M.SIZE - 1 do
    if grid[k] then out[#out + 1] = { k % M.SIZE, k // M.SIZE, grid[k] } end
  end
  return out
end

-- Oxa as it is drawn by default: its first gills, open eyes and a smile.
M.DEFAULT = { "gills_a", "body", "eyes_open", "mouth_smile" }
M.BLINK = { "gills_a", "body", "eyes_closed", "mouth_smile" }

return M
