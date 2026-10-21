# Writes testdata/usd_parts.usdz, the USD reader's test fixture, with
# OpenUSD's Python bindings (pxr):
#     python3 tools/usd_fixture/make.py
# (Arch's pxr may need LD_PRELOAD=/usr/lib/libjemalloc.so.2.)
# What usd_read_test.odin checks: centimeters and Z up; translate,
# rotateXYZ and scale ops, one time-sampled; a prototype (a root `over`)
# referenced twice, one instance binding its own material; a quad and a
# triangle with face-varying indexed st and a materialBind subset; a 5x5
# grid (int arrays long enough to be compressed) and a 32-float array;
# a textured UsdPreviewSurface whose opacity is a texture's red channel
# through a UsdTransform2d; a mesh whose normals' count does not fit their
# interpolation.
import os, struct, zlib, tempfile
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "..", "testdata", "usd_parts.usdz")
work = tempfile.mkdtemp()

def png(path, w, h, rgba):
    raw = b"".join(b"\0" + bytes(rgba[y * w * 4:(y + 1) * w * 4]) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

os.makedirs(os.path.join(work, "tex"))
# 2x2: red's channel r says the opacity (top row 255, bottom 0), green 0.
png(os.path.join(work, "tex", "mask.png"), 2, 2, [255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
png(os.path.join(work, "tex", "color.png"), 2, 2, [10, 20, 30, 255] * 4)

layer = os.path.join(work, "parts.usdc")
stage = Usd.Stage.CreateNew(layer)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 0.01)

red = UsdShade.Material.Define(stage, "/World/Looks/Red")
s = UsdShade.Shader.Define(stage, "/World/Looks/Red/Surface")
s.CreateIdAttr("UsdPreviewSurface")
s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
red.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")

net = UsdShade.Material.Define(stage, "/World/Looks/Net")
s = UsdShade.Shader.Define(stage, "/World/Looks/Net/Surface")
s.CreateIdAttr("UsdPreviewSurface")
reader = UsdShade.Shader.Define(stage, "/World/Looks/Net/St")
reader.CreateIdAttr("UsdPrimvarReader_float2")
reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
xf = UsdShade.Shader.Define(stage, "/World/Looks/Net/Tiling")
xf.CreateIdAttr("UsdTransform2d")
xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2, 3))
xf.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0)
xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.5, 0.25))
for name, f in (("Color", "tex/color.png"), ("Mask", "tex/mask.png")):
    t = UsdShade.Shader.Define(stage, "/World/Looks/Net/" + name)
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f)
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI() if name == "Mask" else reader.ConnectableAPI(), "result")
    t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    t.CreateOutput("r", Sdf.ValueTypeNames.Float)
s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(UsdShade.Shader.Get(stage, "/World/Looks/Net/Color").ConnectableAPI(), "rgb")
s.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(UsdShade.Shader.Get(stage, "/World/Looks/Net/Mask").ConnectableAPI(), "r")
s.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
net.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")

# The prototype: a root over, not drawn itself.
proto = stage.OverridePrim("/Proto")
part = UsdGeom.Mesh.Define(stage, "/Proto/Part")
part.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)])
part.CreateFaceVertexCountsAttr([4, 3])
part.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 2])
st = UsdGeom.PrimvarsAPI(part).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
st.SetIndices([0, 1, 2, 3, 1, 0, 2])
sub = UsdGeom.Subset.CreateGeomSubset(part, "quad", UsdGeom.Tokens.face, [0], "materialBind")
UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(red)

world = UsdGeom.Xform.Define(stage, "/World")
a = UsdGeom.Xform.Define(stage, "/World/A")
a.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
a.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))
a.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
a.GetPrim().GetReferences().AddInternalReference("/Proto")
b = UsdGeom.Xform.Define(stage, "/World/B")
op = b.AddTranslateOp()
op.Set(Gf.Vec3d(0, 50, 0), 1)
op.Set(Gf.Vec3d(0, 99, 0), 2)
b.GetPrim().GetReferences().AddInternalReference("/Proto")
UsdShade.MaterialBindingAPI.Apply(stage.OverridePrim("/World/B/Part")).Bind(net)

grid = UsdGeom.Mesh.Define(stage, "/World/Grid")
grid.CreatePointsAttr([(x, y, 0) for y in range(6) for x in range(6)])
grid.CreateFaceVertexCountsAttr([4] * 25)
grid.CreateFaceVertexIndicesAttr([i for y in range(5) for x in range(5) for i in (y * 6 + x, y * 6 + x + 1, (y + 1) * 6 + x + 1, (y + 1) * 6 + x)])
grid.CreateDisplayColorAttr([(0, 0.5, 0)])
grid.GetPrim().CreateAttribute("weights", Sdf.ValueTypeNames.FloatArray).Set([i * 0.5 for i in range(32)])
# Normals said to be one a face but one a face corner (as the dry dock's
# hull has them): read by their count, per corner. Its corners run 2, 1, 0
# so per point would differ.
odd = UsdGeom.Mesh.Define(stage, "/World/Mislabeled")
odd.CreatePointsAttr([(1, 1, 0), (2, 1, 0), (1, 2, 0)])
odd.CreateFaceVertexCountsAttr([3])
odd.CreateFaceVertexIndicesAttr([2, 1, 0])
odd.CreateNormalsAttr([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
odd.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
stage.GetRootLayer().Save()

if os.path.exists(out):
    os.remove(out)
cwd = os.getcwd()
os.chdir(work)
assert UsdUtils.CreateNewUsdzPackage("parts.usdc", out)
os.chdir(cwd)
print(out, os.path.getsize(out))
