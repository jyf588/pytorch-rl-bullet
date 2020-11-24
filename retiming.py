# (0, b'r_shoulder_out_joint', 0, 7, 6, 1, 0.0, 0.0, -2.07079632679, 1.57079632679, 1000.0, 2000.0, b'r_bicep_link_aux_0', (1.0, 0.0, 0.0), (0.0, -0.066, -0.06), (0.0, 0.0, 0.0, 1.0), -1)
# (1, b'r_shoulder_lift_joint', 0, 8, 7, 1, 0.0, 0.0, -2.07079632679, 1.57079632679, 1000.0, 2000.0, b'r_bicep_link_aux_1', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
# (2, b'r_upper_arm_roll_joint', 0, 9, 8, 1, 0.0, 0.0, -1.57079632679, 1.57079632679, 1000.0, 2000.0, b'r_bicep_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1)
# (3, b'r_elbow_flex_joint', 0, 10, 9, 1, 0.0, 0.0, -3.14159265359, 0.0, 1000.0, 2000.0, b'r_forearm_link_aux', (0.0, 1.0, 0.0), (-0.054, -0.009, -0.2621), (0.0, 0.0, 0.0, 1.0), 2)
# (4, b'r_elbow_roll_joint', 0, 11, 10, 1, 0.0, 0.0, -1.57079632679, 1.57079632679, 1000.0, 2000.0, b'r_forearm_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 3)
# (5, b'r_wrist_roll_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'r_hand_link', (0.0, 0.0, 0.0), (0.0303, -0.00729, -0.302), (1.0, 0.0, 0.0, 1.0341155355510722e-13), 4)
# (6, b'rh_WRJ2', 0, 12, 11, 1, 0.1, 0.0, -1.0471975512, 1.0471975512, 10.0, 20.0, b'rh_wrist', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 5)
# (7, b'rh_WRJ1', 0, 13, 12, 1, 0.1, 0.0, -1.57079632679, 1.57079632679, 30.0, 20.0, b'rh_palm', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0009999999999999974), (0.0, 0.0, 0.0, 1.0), 6)
import inspect
import os
from my_pybullet_envs import utils
import numpy as np

import pybullet as p
import math
import pybullet_utils.bullet_client as bc
from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

# pr = bc.BulletClient(connection_mode=pybullet.DIRECT)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
TimeStep = utils.TS
EPS = np.finfo(float).eps * 4.

arm_dofs = [0, 1, 2, 3, 4, 6, 7]
fin_actdofs = [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]
fin_zerodofs = [8, 13, 18, 24]
fin_tips = [12, 17, 22, 28, 34]
all_findofs = list(np.sort(fin_actdofs+fin_zerodofs))
ee_id = 7
base_init_pos = np.array([-0.30, 0.348, 0.272])
base_init_euler = np.array([0,0,0])

def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    if precise:
        # This code uses a modification of the algorithm described in:
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        # which is itself based on the method described here:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Altered to work with the column vector convention instead of row vectors
        m = M.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]
        q = np.array(q)
        q *= 0.5 / np.sqrt(t)
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """

    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

def pose2mat(pose):
    """
    Convert pose to homogeneous matrix
    :param pose: a (pos, orn) tuple where
    pos is vec3 float cartesian, and
    orn is vec4 float quaternion.
    :return:
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat


def dist_func(a, b, c, t):
    return a * t ** 3 + b * t ** 2 + c * t


def solve_arm_IK(arm_id, sim, w_pos, w_quat, rand_init=False):
    # reset according to wrist 6D pos
    wx_trans = list(w_pos)
    wx_quat = list(w_quat)
    closeEnough = False
    if rand_init:
        sp = list(np.random.uniform(low=-0.03, high=0.03, size=7)) + [0.0]*len(all_findofs)
        sp[3] -= 1.57
    else:
        sp = [-0.44, 0.00, -0.5, -1.8, -0.44, -0.488, -0.8] + [0.0]*len(all_findofs)    # dummy init guess IK
    ll = np.array([sim.getJointInfo(arm_id, i)[8] for i in range(sim.getNumJoints(arm_id))])
    ul = np.array([sim.getJointInfo(arm_id, i)[9] for i in range(sim.getNumJoints(arm_id))])
    ll = ll[arm_dofs+all_findofs]
    ul = ul[arm_dofs+all_findofs]
    jr = ul - ll
    iter = 0
    dist = 1e30
    while not closeEnough and iter < 50:
        for ind in range(len(arm_dofs)):
            sim.resetJointState(arm_id, arm_dofs[ind], sp[ind])

        jointPoses = sim.calculateInverseKinematics(arm_id, ee_id, wx_trans, wx_quat,
                                                  lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                  jointRanges=jr.tolist(),
                                                  restPoses=sp)
        # jointPoses = self.sim.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat)

        sp = np.array(jointPoses)[range(7)].tolist()
        # print(sp)

        wx_now = sim.getLinkState(arm_id, ee_id)[4]
        dist = np.linalg.norm(np.array(wx_now) - np.array(wx_trans))
        # print("dist=", dist)
        if dist < 1e-3: closeEnough = True
        iter += 1
    if dist > 1e-3: sp = None     # TODO
    return sp

def Hermite_Curve(P1, P4, R1, R4, T):
    """
    T: the number of points in the trajectory. (P1 and P4 included)
    """
    Mh = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
    Gh = np.concatenate((P1, P4, R1, R4), axis=0).reshape(4, -1)
    t = np.arange(T) / (T - 1)
    t = t.reshape(T, 1)
    Th = np.concatenate((t**3, t**2, t, np.ones((T, 1))), axis=1)
    curve = np.dot(np.dot(Th, Mh), Gh)
    return curve

def update_traj(Traj, obj_pos, arm_dofs, dq0=7*[0], v1=0, HC_v=0.4):
    sim = bc.BulletClient(connection_mode=p.DIRECT)
    # robot = InmoovShadowNew(
    #     init_noise=init_noise,
    #     timestep=TimeStep,
    #     np_random=np_random,
    #     sim=sim
    # )
    sim.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
    sim.setPhysicsEngineParameter(deterministicOverlappingPairs=0)
    sim.setTimeStep(utils.TS)
    sim.setGravity(0, 0, -utils.GRAVITY)
    arm_id = sim.loadURDF(os.path.join(currentdir,
                                             "my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf"),
                                 list(base_init_pos), sim.getQuaternionFromEuler(list(base_init_euler)),
                                 flags=sim.URDF_USE_SELF_COLLISION | sim.URDF_USE_INERTIA_FROM_FILE
                                       | sim.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                 useFixedBase=1)
    HC_ppt = 0.1
    Traj = Traj[:, 0:7]
    dist_list = [0]  # distance at each timestepa
    ini_armq = Traj[0]
    # numJoints = pr.getNumJoints(arm_id)
    for i in range(7):
        sim.resetJointState(arm_id, arm_dofs[i], ini_armq[i], dq0[i])
    sim.stepSimulation()
    hand_pos, _, _, _, _, _, v0, _ = sim.getLinkState(arm_id, 7, computeLinkVelocity=1, computeForwardKinematics=1)
    v0 = np.linalg.norm(v0)
    for ind in range(1, len(Traj)):
        tar_armq = Traj[ind]
        for i in range(7):
            sim.resetJointState(arm_id, arm_dofs[i], tar_armq[i])
        sim.stepSimulation()
        hand_pos_prime = sim.getLinkState(arm_id, 7, computeForwardKinematics=1)[0]
        dist_list.append(np.linalg.norm(np.array(hand_pos_prime) - np.array(hand_pos)))
        hand_pos = hand_pos_prime

    _, hand_quat, localInertialFramePos, localInertialFrameOri, _, _, _, _ = sim.getLinkState(arm_id, 7, computeLinkVelocity=1, computeForwardKinematics=1)
    # data = pr.getLinkState(arm_id, 7, computeLinkVelocity=1, computeForwardKinematics=1)
    # hand_quat = pr.getLinkState(arm_id, 7, computeLinkVelocity=1, computeForwardKinematics=1)[1]


    dist = np.cumsum(np.array(dist_list))

    dist_ppt = dist / dist[-1]
    idx = np.searchsorted(dist_ppt, 1 - HC_ppt)
    # dist_thresh = 0.3


    P1 = Traj[idx]
    P4 = Traj[-1]
    T = len(Traj) - idx
    R1 = ((Traj[idx + 1] - Traj[idx]) * v1 * TimeStep / (dist[idx + 1] - dist[idx])) * (T - 1)
    virtual_pos = np.array(hand_pos) + 0.2 * (obj_pos - np.array(hand_pos))

    comLinkFrame = pose2mat((virtual_pos, hand_quat))
    localInertialFrame = pose2mat((localInertialFramePos, localInertialFrameOri))
    urdfLinkFrame = comLinkFrame.dot(np.linalg.inv(localInertialFrame))
    ik_pos = urdfLinkFrame[:3, -1]
    ik_ori = mat2quat(urdfLinkFrame[:3, :3])

    Qdst = solve_arm_IK(arm_id, sim, ik_pos, ik_ori)
    assert Qdst is not None
    HC_diff_q = np.array(Qdst) - P4
    assert v1 > 0
    HC_t = np.linalg.norm(virtual_pos - np.array(hand_pos)) / HC_v
    HC_final_dq = HC_diff_q / HC_t
    R4 = HC_final_dq * TimeStep * (T - 1)
    HC_traj = np.zeros((len(Traj), 7))
    HC_traj[:idx] = Traj[:idx].copy()
    HC_traj[idx:] = Hermite_Curve(P1=P1, P4=P4, R1=R1, R4=R4, T=T)
    HC_dist_list = dist_list[:idx+1]

    for i in range(7):
        sim.resetJointState(arm_id, arm_dofs[i], P1[i], 0)
    sim.stepSimulation()
    hand_pos = sim.getLinkState(arm_id, 7, computeLinkVelocity=1, computeForwardKinematics=1)[0]

    for ind in range(idx+1, len(Traj)):
        tar_armq = HC_traj[ind]
        for i in range(7):
            sim.resetJointState(arm_id, arm_dofs[i], tar_armq[i])
        sim.stepSimulation()
        hand_pos_prime = sim.getLinkState(arm_id, 7, computeForwardKinematics=1)[0]
        HC_dist_list.append(np.linalg.norm(np.array(hand_pos_prime) - np.array(hand_pos)))
        hand_pos = hand_pos_prime
    HC_dist = np.cumsum(np.array(HC_dist_list))

    T = len(Traj) * TimeStep
    a = (v0 + v1) / (T ** 2) - 2 * HC_dist[-1] / (T ** 3)
    b = 3 * HC_dist[-1] / (T ** 2) - (2 * v0 + v1) / T
    c = v0

    new_traj = np.zeros((Traj.shape[0], 7))
    new_traj[0] = Traj[0]

    for t in range(1, len(Traj)):
        ts = t * TimeStep
        dist_targ = dist_func(a, b, c, ts)
        idx = np.searchsorted(HC_dist, dist_targ)
        dist0 = HC_dist[idx - 1]
        dist1 = HC_dist[idx]
        ppt = (dist_targ - dist0) / (dist1 - dist0)
        new_traj[t] = HC_traj[idx - 1] + ppt * (HC_traj[idx] - HC_traj[idx - 1])

    # for i in range(7):
    #     p.resetJointState(robot.arm_id, arm_dofs[i], ini_armq[i], dq0[i])
    sim.disconnect()
    return new_traj


