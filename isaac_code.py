# Isaac Sim 4.5+ demo: a 50‑link capsule “rope” hung between two cubes
#
# Paste into Window → Script Editor and press Run, then click Play in the stage.

from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd
from omni.physx.scripts import utils

stage = omni.usd.get_context().get_stage()

# ─── parameters ──────────────────────────────────────────────────────────────
NUM_LINKS      = 50
RADIUS         = 0.025      # capsule radius  in m
HEIGHT         = 0.025      # cylinder length in m  (full, not half)
GAP            = 0.002      # clearance between links
SPACING        = HEIGHT + 2*RADIUS + GAP
Z0             = 0.0        # height of the whole chain

CHAIN_ROOT     = "/World/Chain"
ANCHOR0_PATH   = "/World/Cube"
ANCHOR1_PATH   = "/World/Cube_2"
PHYS_SCENE     = "/World/physicsScene"
# ─────────────────────────────────────────────────────────────────────────────

# physics scene (normal Earth gravity)
scene = UsdPhysics.Scene.Define(stage, PHYS_SCENE)
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# helper: static / kinematic cube prim
def make_anchor(path: str, x_pos: float):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(0.06)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3f(x_pos, 0, Z0))

    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb.CreateKinematicEnabledAttr().Set(True)             # fixed in space
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()

# create two end cubes
anchor0 = make_anchor(ANCHOR0_PATH, -SPACING)
anchor1 = make_anchor(ANCHOR1_PATH,  NUM_LINKS*SPACING)

# container Xform for the whole chain
stage.DefinePrim(CHAIN_ROOT, "Xform")

prev_prim = anchor0                                              # start link‑to‑cube

for i in range(NUM_LINKS):
    link_path = f"{CHAIN_ROOT}/Link_{i}"
    link     = stage.DefinePrim(link_path, "Capsule")

    # capsule dims
    link.GetAttribute("radius").Set(RADIUS)
    link.GetAttribute("height").Set(HEIGHT)

    # pose: translate along +X, rotate 90° about Y so the long axis ≡ X
    xf = UsdGeom.Xformable(link)
    xf.AddTranslateOp().Set(Gf.Vec3f(i*SPACING, 0, Z0))
    xf.AddOrientOp().Set(Gf.Quatf(0.7071068, Gf.Vec3f(0, 0.7071068, 0)))

    # rigid‑body + collider
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.CollisionAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr().Set(0.5)

    # D6 joint to previous element
    joint_path = f"{CHAIN_ROOT}/Joint_{i}"
    joint = utils.createJoint(stage, "D6Joint",
                            prev_prim, link,
                            path=joint_path) 

    # lock translation; leave twist (rotX) free, limit swing (rotY, rotZ)
    for axis in ["transX", "transY", "transZ"]:
        lock = UsdPhysics.LimitAPI.Apply(joint, axis)
        lock.CreateLowAttr().Set( 1.0)          # low > high ⇒ locked
        lock.CreateHighAttr().Set(-1.0)

    for axis in ["rotY", "rotZ"]:
        swing = UsdPhysics.LimitAPI.Apply(joint, axis)
        swing.CreateLowAttr().Set(-0.3)         # ±17°
        swing.CreateHighAttr().Set( 0.3)

    prev_prim = link                                        # next link attaches here

# final link → right‑hand cube
utils.createJoint(stage, "D6Joint",
                  prev_prim, anchor1,
                  jointPrimPath=f"{CHAIN_ROOT}/FinalJoint")

print("▶ Chain created (two anchored cubes, 50 capsule links). Press Play!")
