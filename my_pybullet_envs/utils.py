import numpy as np
import pybullet as p
import os
import pickle
from typing import *

PLACE_START_CLEARANCE = 0.14

PALM_POS_OF_INIT = [-0.18, 0.095, 0.11]
PALM_EULER_OF_INIT = [1.8, -1.57, 0]

SHAPE_IND_MAP = {-1: p.GEOM_SPHERE, 0: p.GEOM_CYLINDER, 1: p.GEOM_BOX}
SHAPE_NAME_MAP = {
    "sphere": p.GEOM_SPHERE,
    "cylinder": p.GEOM_CYLINDER,
    "box": p.GEOM_BOX,
}
GEOM2SHAPE = {geom: shape for shape, geom in SHAPE_NAME_MAP.items()}
# only used for construct GT scene, language should be more forgiving wrt shape name, see lang module.

COLOR2RGBA = {
    "red": [0.8, 0.0, 0.0, 1.0],
    "grey": [0.4, 0.4, 0.4, 1.0],
    "yellow": [0.8, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 0.8, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
}
RGBA2COLOR = {tuple(rgba): color for color, rgba in COLOR2RGBA.items()}

# TODO: should use a config file? command line change?
MASS_MIN = 1.0
MASS_MAX = 5.0
MU_MIN = 0.8
MU_MAX = 1.2
HALF_W_MIN = 0.03
HALF_W_MAX = 0.05
HALF_W_MIN_BTM = 0.045  # only stack on larger objects
H_MIN = 0.13
H_MAX = 0.18

TX_MIN = -0.1
TX_MAX = 0.25
TY_MIN = -0.1
TY_MAX = 0.5

TABLE_OFFSET = [0.1, 0.2, 0.0]
# TODO: during training, make table a bit thicker/higher?

BULLET_CONTACT_ITER = 200

INIT_PALM_CANDIDATE_ANGLES = [
    0.0,
    3.14 / 4,
    6.28 / 4,
    9.42 / 4,
    3.14,
    -9.42 / 4,
    -6.28 / 4,
    -3.14 / 4,
]
INIT_PALM_CANDIDATE_QUATS = [
    p.getQuaternionFromEuler([0, 0, cand_angle])
    for cand_angle in INIT_PALM_CANDIDATE_ANGLES
]


def perturb(np_rand_gen, arr, r=0.02):
    r = np.abs(r)
    return np.copy(
        np.array(arr) + np_rand_gen.uniform(low=-r, high=r, size=len(arr))
    )


def perturb_scalar(np_rand_gen, num, r=0.02):
    r = np.abs(r)
    return num + np_rand_gen.uniform(low=-r, high=r)


def sample_tx_ty_tz(
    np_rand_gen, is_universal_xy, is_on_floor, xy_noise, z_noise
):
    # tx ty tz can be out of arm reach
    # tx ty used for planning, and are part of the robot obs
    # tx_act, ty_act are the actual btm obj x y
    # tz_act is the actual bottom obj height
    # tz used for planning and robot obs

    if is_universal_xy:
        tx = np_rand_gen.uniform(low=TX_MIN, high=TX_MAX)
        ty = np_rand_gen.uniform(low=TY_MIN, high=TY_MAX)
    else:
        tx = 0.0
        ty = 0.0

    tx_act = perturb_scalar(np_rand_gen, tx, xy_noise)
    ty_act = perturb_scalar(np_rand_gen, ty, xy_noise)

    if is_on_floor:
        tz_act = 0
        tz = 0
    else:
        tz_act = np_rand_gen.uniform(H_MIN, H_MAX)
        tz = perturb_scalar(np_rand_gen, tz_act, z_noise)

    return tx, ty, tz, tx_act, ty_act, tz_act


def create_prim_shape(
    mass,
    shape,
    dim,
    mu=1.0,
    init_xyz=(0, 0, 0),
    init_quat=(0, 0, 0, 1),
    color=(0.9, 0.9, 0.9, 1),
):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) vec1 for sphere
    # init_xyz vec3 of obj location
    visual_shape_id = None
    collision_shape_id = None
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim)
        collision_shape_id = p.createCollisionShape(
            shapeType=shape, halfExtents=dim
        )
    elif shape == p.GEOM_CYLINDER:
        # visual_shape_id = p.createVisualShape(shapeType=shape, radius=dim[0], length=dim[1])
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1])
        # collision_shape_id = p.createCollisionShape(shapeType=shape, radius=dim[0], length=dim[1])
        collision_shape_id = p.createCollisionShape(
            shape, dim[0], [1, 1, 1], dim[1]
        )
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0])
        collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=init_xyz,
        baseOrientation=init_quat,
    )

    p.changeVisualShape(sid, -1, rgbaColor=color)

    p.changeDynamics(sid, -1, lateralFriction=mu)

    return sid


def create_sym_prim_shape_helper(
    odict, init_xyz=(0, 0, 0), init_quat=(0, 0, 0, 1)
):
    # NOTE: half_width ignored for spheres. if shape is sphere, must pass in None as half_width
    # the input odict is a dict of obj metadata.

    shape = odict["shape"]
    dim = to_bullet_dimension(shape, odict["half_width"], odict["height"])
    if "color" in odict:
        sid = create_prim_shape(
            odict["mass"],
            shape,
            dim,
            odict["mu"],
            init_xyz,
            init_quat,
            odict["color"],
        )
    else:
        sid = create_prim_shape(
            odict["mass"],
            shape,
            dim,
            odict["mu"],
            init_xyz,
            init_quat,
            (0.9, 0.9, 0.9, 1),
        )
        # give some default white color.
    return sid


def to_bullet_dimension(shape, half_width, height):
    # convert half-width and height for our symmetrical primitives to bullet dimension def
    # NOTE: half_width ignored for spheres. if shape is sphere, must pass in None as half_width
    dim = None
    if shape == p.GEOM_BOX:
        dim = [half_width, half_width, height / 2.0]
    elif shape == p.GEOM_CYLINDER:
        dim = [half_width, height]
    elif shape == p.GEOM_SPHERE:
        assert half_width is None
        dim = [height / 2.0]
    return dim


def from_bullet_dimension(shape, dim):
    # the inverse func of above,
    half_width = None
    height = None
    if shape == p.GEOM_BOX:
        assert dim[0] == dim[1]
        half_width, height = dim[0], dim[2] * 2.0
    elif shape == p.GEOM_CYLINDER:
        half_width, height = dim[0], dim[1]
    elif shape == p.GEOM_SPHERE:
        height = dim[0] * 2.0
    return half_width, height


def get_n_optimal_init_arm_qs(
    robot, p_pos_of, p_quat_of, desired_obj_pos, table_id, n=2, wrist_gain=1.0
):
    # NOTE: robot is a InMoov object
    # NOTE: if table_id none, do not check init arm collision with table.
    arm_qs_costs = []
    ref = np.array([0.0] * 3 + [-1.57] + [0.0] * 3)
    for ind, cand_quat in enumerate(INIT_PALM_CANDIDATE_QUATS):
        # p_pos_of_ave, p_quat_of_ave = p.invertTransform(o_pos_pf, o_quat_pf)
        p_pos, p_quat = p.multiplyTransforms(
            desired_obj_pos, cand_quat, p_pos_of, p_quat_of
        )
        cand_arm_q = robot.solve_arm_IK(p_pos, p_quat)
        if cand_arm_q is not None:
            cps = []
            if table_id is not None:
                p.stepSimulation()  # TODO
                cps = p.getContactPoints(bodyA=robot.arm_id, bodyB=table_id)
            if len(cps) == 0:
                diff = np.array(cand_arm_q) - ref
                diff[-1] *= wrist_gain
                cand_cost = np.sum(np.abs(diff))  # changed to l1
                arm_qs_costs.append((cand_arm_q, cand_cost))
    arm_qs_costs_sorted = sorted(arm_qs_costs, key=lambda x: x[1])[
        :n
    ]  # fine if length < n
    return [arm_q_cost[0] for arm_q_cost in arm_qs_costs_sorted]


def obj_pos_and_upv_to_obs(o_pos, o_upv, tx, ty):
    objObs = []
    o_pos = np.array(o_pos)
    o_pos -= [tx, ty, 0]
    o_pos = o_pos * 3.0
    objObs.extend(o_pos)
    objObs.extend(o_upv)
    return objObs


def quat_to_upv(quat):
    rotmat = np.array(p.getMatrixFromQuaternion(quat))
    upv = [rotmat[2], rotmat[5], rotmat[8]]
    return upv


def save_pickle(path: str, data: Any):
    """
    Args:
        path: The path of the pickle file to save.
        data: The data to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# def get_pos_upv_height_from_obj(use_vision, odict):
#     if odict['id']


# looks like need two dicts for info of two objs.
# --> do we need a class?
# --> do we want to store the current 6D of objects? And update
# will probably still get obj current 6D using obj_ids and p.
# just a dict of obj_dict['id'], ['mass'], ['shape'], (['dim']) ['color'], ['friction'], ['half_width'], ['height']
# small class might be useful

# need a function getCurrentState, so state saver can be deprecated. (no need to put track obj etc.)
# visionObj class and visionInfer class still need to be added to env code.
# the input shall be a list of obj metadata *dicts*, and robot arm id.
# save pickle does not seem necessary. why?

# obj6DtoObs_UpVec(
# obj_pos_and_upv_to_obs in demo env.

# predict() currently takes a list of BULLET o_ids, which means vision prediction has BULLET id as field.
# it should instead always output the predictions of ALL objects with ids 0..N-1
# How do we know which xyz is what we want? indexed by language module.
# easy fix: init vision module should be fine even if each obj segmentation does not have bullet id.
# stacking vision always output top obj 6D followed by btm obj so no indexing is needed.


# TODO: there are two tables in demo...

# shape,color,init_position
# bullet_id
# radius, height, pos, quat

# multiple dicts
# michelle uses a dict of dicts with oids as key
# there is also prediction dicts from vision output (position, up_vector, height) * 2 for stacking;
# (shape,color,init_position) for init planning
# GT scene description/language has shape,color,init_position that is passed to Bullet to create scene
# GT scene other info

# Eventually, lang need to take in vision predicted shape,color,init_position
# list, without oid as keys.

# should split GT info dict & prediction dict

# are we predicting (shape,color) for stacking still?

