import numpy as np
import pybullet as p


SHAPE_IND_MAP = {-1: p.GEOM_SPHERE, 0: p.GEOM_CYLINDER, 1: p.GEOM_BOX}
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
TX_MAX = 0.3
TY_MIN = -0.15
TY_MAX = 0.55

TABLE_OFFSET = [0.1, 0.2, 0.0]

BULLET_CONTACT_ITER = 200

def perturb(np_rand_gen, arr, r=0.02):
    r = np.abs(r)
    return np.copy(np.array(arr) + np_rand_gen.uniform(low=-r, high=r, size=len(arr)))


def perturb_scalar(np_rand_gen, num, r=0.02):
    r = np.abs(r)
    return num + np_rand_gen.uniform(low=-r, high=r)


def create_prim_shape(mass, shape, dim, mu=1.0, init_xyz=(0, 0, 0), init_quat=(0, 0, 0, 1), color=(0.6, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) vec1 for sphere
    # init_xyz vec3 of obj location
    visual_shape_id = None
    collision_shape_id = None
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim)
        collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        # visual_shape_id = p.createVisualShape(shapeType=shape, radius=dim[0], length=dim[1])
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1])
        # collision_shape_id = p.createCollisionShape(shapeType=shape, radius=dim[0], length=dim[1])
        collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0])
        collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=init_xyz, baseOrientation=init_quat)

    p.changeVisualShape(sid, -1, rgbaColor=color)

    p.changeDynamics(sid, -1, lateralFriction=mu)

    return sid


def create_sym_prim_shape_helper(odict, init_xyz=(0, 0, 0),
                          init_quat=(0, 0, 0, 1), color=(0.6, 0, 0, 1)):
    # NOTE: half_width ignored for spheres. if shape is sphere, must pass in None as half_width
    # the input odict is a dict of obj metadata.

    shape = odict['shape']
    dim = to_bullet_dimension(shape, odict['half_width'], odict['height'])
    sid = create_prim_shape(odict['mass'], shape, dim, odict['mu'], init_xyz, init_quat, color)
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
        half_width, height = dim[0], dim[2]*2.0
    elif shape == p.GEOM_CYLINDER:
        half_width, height = dim[0], dim[1]
    elif shape == p.GEOM_SPHERE:
        height = dim[0]*2.0
    return half_width, height


# looks like need two dicts for info of two objs.
# --> do we need a class?
# --> do we want to store the current 6D of objects? And update
# will probably still get obj current 6D using obj_ids and p.
# just a dict of obj_dict['id'], ['mass'], ['shape'], (['dim']) ['color'], ['friction'], ['half_width'], ['height']
# small class might be useful

# need a function getCurrentState, so state saver can be deprecated.
# save pickle does not seem necessary. why?

# obj6DtoObs_UpVec(
# obj_pos_and_up_to_obs in demo env.
