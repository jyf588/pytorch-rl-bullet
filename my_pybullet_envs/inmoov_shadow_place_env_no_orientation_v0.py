from .inmoov_shadow_hand_v2_no_orientation import InmoovShadowNew

from . import utils

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
from state_saver import StateSaver

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


# multiple placing init pose & place on floor. / remove collision with floor init.

# easier way:
# 1. load floor first
# 2. permute the list of arm candidates  (some IK infeasible, filtered out)
# 3. reset arm (hand using init grasp pose)
# 4. if collide with floor, filter out, return a list (max 2) of arm q ranked by optimality
# If none exist, skip this txty/sample.
# init_done = False
# while not init_done:


class InmoovShadowHandPlaceEnvNoOrientationV0(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(
        self,
        renders=False,
        init_noise=True,  # variation during reset
        up=True,
        random_top_shape=True,  # TODO: deprecate these 2 flags
        det_top_shape_ind=1,  # if not random shape, 1 means always box
        cotrain_stack_place=True,
        place_floor=True,  # if not cotrain, is stack or place-on-floor
        grasp_pi_name=None,
        exclude_hard=False,
        use_gt_6d=True,
        gt_only_init=False,
        vision_skip=2,
        control_skip=6,
        obs_noise=False,  # noisy (imperfect) observation
        n_best_cand=2,
        use_obj_heights=False,
        save_states=False,
        states_dir=None,
        with_stack_place_bit=True,
    ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self.cotrain_stack_place = cotrain_stack_place
        self.place_floor = place_floor
        self.exclude_hard = exclude_hard
        self.hard_orn_thres = 0.85

        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init

        self.obs_noise = obs_noise

        self.vision_skip = vision_skip
        self.vision_counter = 0

        self.n_best_cand = int(n_best_cand)

        self.with_stack_place_bit = with_stack_place_bit

        self.use_obj_heights = (
            use_obj_heights  # use or not obj heights info in planning and obs
        )

        self.top_id = None
        self.btm_id = None
        self.objs = {}  # btm obj dict and top obj dict indexed by bullet id
        self.table_id = None

        # specify from command line, otherwise use default
        if grasp_pi_name:
            self.grasp_pi_name = grasp_pi_name
        else:
            if not random_top_shape:
                if det_top_shape_ind:
                    self.grasp_pi_name = "0311_box_2_n_20_50"
                else:
                    self.grasp_pi_name = "0311_cyl_2_n_20_50"
            else:
                # self.grasp_pi_name = "0313_2_n_25_45"
                # self.grasp_pi_name = "0325_graspco_16_n_w0_25_45"
                self.grasp_pi_name = "0331_co_2_w_25_45"

        self.start_clearance = utils.PLACE_START_CLEARANCE

        self._timeStep = 1.0 / 240.0
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.np_random = None
        self.robot = None
        self.seed(0)  # used once temporarily, will be overwritten outside by env
        self.viewer = None
        self.timer = 0
        self.trial = 0

        self.control_skip = int(control_skip)
        # shadow hand is 22-5=17dof
        self.action_scale = np.array(
            [0.009 / self.control_skip] * 7 + [0.024 / self.control_skip] * 17
        )

        self.tx = -1  # dummy
        self.ty = -1  # dummy
        self.tz = -1  # dummy
        self.tx_act = -1  # dummy
        self.ty_act = -1  # dummy
        self.tz_act = -1  # dummy

        self.desired_obj_pos_final = None

        (
            self.o_pos_pf_ave,
            self.o_quat_pf_ave,
            self.init_states,
        ) = utils.read_grasp_final_states_from_pickle(self.grasp_pi_name)

        # print(self.o_pos_pf_ave)
        # print(self.o_quat_pf_ave)
        # print(self.init_states[10])
        # print(self.init_states[51])
        # print(self.init_states[89])

        # Instantiate a state saver if requested.
        self.save_states = save_states
        if self.save_states:
            self.state_saver = StateSaver(out_dir=states_dir)

        self.reset()  # and update init obs

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * action_dim), high=np.array([+1.0] * action_dim),
        )
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567] * obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * obs_dummy, high=np.inf * obs_dummy
        )

    def reset_robot_top_object_from_sample(self, arm_q):
        while True:
            ran_ind = int(
                self.np_random.uniform(low=0, high=len(self.init_states) - 0.1)
            )
            state = self.init_states[ran_ind]

            o_pos_pf = state["obj_pos_in_palm"]
            o_quat_pf = state["obj_quat_in_palm"]
            if self.init_noise:
                o_pos_pf = list(utils.perturb(self.np_random, o_pos_pf, 0.005))
                o_quat_pf = list(utils.perturb(self.np_random, o_quat_pf, 0.005))
            all_fin_q_init = state["all_fin_q"]
            # all_fin_q_init = utils.perturb(self.np_random, all_fin_q_init, 0.02)
            tar_fin_q_init = state["fin_tar_q"]

            self.robot.reset_with_certain_arm_q_finger_states(
                arm_q, all_fin_q_init, tar_fin_q_init
            )

            p_pos, p_quat = self.robot.get_link_pos_quat(self.robot.ee_id)
            o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

            z_axis, _ = p.multiplyTransforms(
                [0, 0, 0], o_quat, [0, 0, 1], [0, 0, 0, 1]
            )  # R_cl * unitz[0,0,1]
            rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))
            # print(rotMetric, rotMetric)
            if self.exclude_hard and rot_metric < self.hard_orn_thres:
                continue
            else:
                top_dict = {
                    "shape": utils.GEOM2SHAPE[state["obj_shape"]],
                    "mass": self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX),
                    "mu": self.np_random.uniform(utils.MU_MIN, utils.MU_MAX),
                    "position": o_pos,
                    "orientation": o_quat,
                    "last_position": o_pos,
                    "last_orientation": o_quat,
                    "radius": utils.from_bullet_dimension(
                        state["obj_shape"], state["obj_dim"]
                    )[0],
                    "height": utils.from_bullet_dimension(
                        state["obj_shape"], state["obj_dim"]
                    )[1],
                }
                self.top_id = utils.create_sym_prim_shape_helper_new(top_dict)
                self.objs[self.top_id] = top_dict

                # only used for reward calc.
                self.desired_obj_pos_final = [
                    self.tx_act,
                    self.ty_act,
                    top_dict["height"] / 2.0 + self.tz_act,
                ]
                return

    def sample_valid_arm_q(self):
        while True:
            if self.init_noise:
                # TODO: noise large enough?
                (
                    self.tx,
                    self.ty,
                    self.tz,
                    self.tx_act,
                    self.ty_act,
                    self.tz_act,
                ) = utils.sample_tx_ty_tz(
                    self.np_random, self.up, self.place_floor, 0.015, 0.02
                )
            else:
                (
                    self.tx,
                    self.ty,
                    self.tz,
                    self.tx_act,
                    self.ty_act,
                    self.tz_act,
                ) = utils.sample_tx_ty_tz(
                    self.np_random, self.up, self.place_floor, 0.0, 0.0
                )

            # TODO: this will affect demo env
            if self.use_obj_heights:
                desired_obj_pos = [
                    self.tx,
                    self.ty,
                    self.start_clearance + self.tz,
                ]  # used for planning
            else:
                if self.place_floor:
                    desired_obj_pos = [
                        self.tx,
                        self.ty,
                        utils.perturb_scalar(
                            self.np_random, self.start_clearance + 0.0, 0.01
                        ),
                    ]  # used for planning
                else:
                    desired_obj_pos = [
                        self.tx,
                        self.ty,
                        utils.perturb_scalar(
                            self.np_random, self.start_clearance + utils.H_MAX, 0.01,
                        ),  # always start from higher
                    ]  # used for planning

            p_pos_of_ave, p_quat_of_ave = p.invertTransform(
                self.o_pos_pf_ave, self.o_quat_pf_ave
            )

            if "sph" in self.grasp_pi_name:  # TODO: hardcoded
                _, desired_obj_quat = p.multiplyTransforms(
                    [0, 0, 0],
                    p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
                    [0, 0, 0],
                    self.o_quat_pf_ave,
                )
            else:
                desired_obj_quat = [0.0, 0, 0, 1]  # box or cyl be upright

            arm_qs = utils.get_n_optimal_init_arm_qs(
                self.robot,
                p_pos_of_ave,
                p_quat_of_ave,
                desired_obj_pos,
                self.table_id,
                n=self.n_best_cand,
                desired_obj_quat=desired_obj_quat,
            )
            if len(arm_qs) == 0:
                continue
            else:
                arm_q = arm_qs[self.np_random.randint(len(arm_qs))]
                return arm_q

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0
        self.vision_counter = 0
        self.objs = {}

        if self.cotrain_stack_place:
            self.place_floor = self.np_random.randint(10) >= 7  # 30%

        mu_f = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        self.table_id = utils.create_table(mu_f)

        self.robot = InmoovShadowNew(
            init_noise=False, timestep=self._timeStep, np_random=self.np_random
        )

        arm_q = self.sample_valid_arm_q()  # reset done during solving IK

        if not self.place_floor:
            btm_xyz = [self.tx_act, self.ty_act, self.tz_act / 2.0]
            btm_quat = p.getQuaternionFromEuler(
                [0.0, 0.0, self.np_random.uniform(low=0, high=2.0 * math.pi)]
            )
            btm_dict = {
                "shape": utils.SHAPE_IND_TO_NAME_MAP[
                    np.random.randint(2)
                ],  # btm cyl or box
                "mass": self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX),
                "mu": self.np_random.uniform(utils.MU_MIN, utils.MU_MAX),
                "position": btm_xyz,
                "orientation": btm_quat,
                "last_position": btm_xyz,
                "last_orientation": btm_quat,
                "radius": self.np_random.uniform(
                    utils.HALF_W_MIN_BTM, utils.HALF_W_MAX
                ),
                "height": self.tz_act,
            }
            if btm_dict["shape"] == "box":
                btm_dict["radius"] *= 0.8

            self.btm_id = utils.create_sym_prim_shape_helper_new(btm_dict)
            self.objs[self.btm_id] = btm_dict

        self.reset_robot_top_object_from_sample(arm_q)

        p.stepSimulation()  # TODO

        self.observation = self.getExtendedObservation()

        if self.save_states:
            task = "place" if self.place_floor else "stack"
            self.state_saver.track(
                trial=self.trial,
                task=task,
                tx_act=self.tx_act,
                ty_act=self.ty_act,
                tx=self.tx,
                ty=self.ty,
                odicts=self.objs,
                robot_id=self.robot.arm_id,
            )
        self.trial += 1
        return np.array(self.observation)

    def step(self, action):
        for _ in range(self.control_skip):
            # action is not in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        # if upright, reward
        # if use small force, reward

        reward = 0.0
        top_pos, top_quat = p.getBasePositionAndOrientation(self.top_id)
        top_vels = p.getBaseVelocity(self.top_id)
        top_lin_v = np.array(top_vels[0])
        top_ang_v = np.array(top_vels[1])

        # we only care about the upright(z) direction
        z_axis, _ = p.multiplyTransforms(
            [0, 0, 0], top_quat, [0, 0, 1], [0, 0, 0, 1]
        )  # R_cl * unitz[0,0,1]
        rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))

        xyz_metric = 1 - (
            np.minimum(
                np.linalg.norm(
                    np.array(self.desired_obj_pos_final) - np.array(top_pos)
                ),
                0.15,
            )
            / 0.15
        )
        lin_v_r = np.linalg.norm(top_lin_v)
        # print("lin_v", lin_v_r)
        ang_v_r = np.linalg.norm(top_ang_v)
        # print("ang_v", ang_v_r)
        vel_metric = 1 - np.minimum(lin_v_r * 4.0 + ang_v_r, 5.0) / 5.0

        reward += np.maximum(rot_metric * 20 - 15, 0.0)
        # print(np.maximum(rot_metric * 20 - 15, 0.))
        reward += xyz_metric * 5
        # print(xyz_metric * 5)
        reward += vel_metric * 5
        # print(vel_metric * 5)
        # print("upright", reward)

        diff_norm = self.robot.get_norm_diff_tar()  # TODO: necessary?
        reward += 10.0 / (diff_norm + 1.0)
        # # print(10. / (diff_norm + 1.))

        any_hand_contact = False
        hand_r = 0
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
            con_this_link = False
            for cp in cps:
                if cp[1] != cp[2]:  # not self-collision of the robot
                    con_this_link = True
                    break
            if con_this_link:
                any_hand_contact = True
            else:
                hand_r += 0.5
        reward += hand_r - 7
        # print("no contact", hand_r - 7.0)

        reward -= self.robot.get_4_finger_deviation() * 0.3

        #
        # if self.timer == 99 * self.control_skip:
        #     print(rot_metric, xyz_metric, vel_metric, any_hand_contact)
        #     # print(any_hand_contact)

        if (
            rot_metric > 0.9
            and xyz_metric > 0.6
            and vel_metric > 0.6
            # and meaningful_c
        ):  # close to placing
            reward += 5.0
            # print("upright")
            if not any_hand_contact:
                reward += 20.0
                # print("no hand con")

        # print("r_total", reward)

        obs = self.getExtendedObservation()

        if self.save_states:
            self.state_saver.save_state()

        return obs, reward, False, {}

    def obj6DtoObs_UpVec(self, o_pos, o_orn, is_sph=False):
        o_pos = np.array(o_pos)
        if is_sph:
            o_orn = [0.0, 0, 0, 1]
        o_upv = utils.quat_to_upv(o_orn)

        if self.obs_noise:
            o_pos = utils.perturb(self.np_random, o_pos, r=0.02)
            o_upv = utils.perturb(self.np_random, o_upv, r=0.03)
            obj_obs = utils.obj_pos_and_upv_to_obs(o_pos, o_upv, self.tx, self.ty)
        else:
            o_pos = o_pos
            o_upv = o_upv
            obj_obs = utils.obj_pos_and_upv_to_obs(
                o_pos, o_upv, self.tx_act, self.ty_act
            )

        return obj_obs

    def update_delayed_obj_6d(self, obj_id):
        # TODO: Note! obj['position'] does not get updated per-frame!
        # maybe per-frame obj['position'] is never needed, even for logging a rough obj['position'] will be enough
        self.objs[obj_id]["last_position"] = self.objs[obj_id]["position"]
        self.objs[obj_id]["last_orientation"] = self.objs[obj_id]["orientation"]

        (
            self.objs[obj_id]["position"],
            self.objs[obj_id]["orientation"],
        ) = p.getBasePositionAndOrientation(obj_id)

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation(diff_tar=True)

        curContact = []
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
            con_this_link = False
            for cp in cps:
                if cp[1] != cp[2]:  # not self-collision of the robot
                    con_this_link = True
                    break
            if con_this_link:
                curContact.extend([1.0])
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        if self.use_obj_heights:
            xyz = np.array([self.tx, self.ty, self.tz])
            self.observation.extend(list(xyz))
            if self.obs_noise:
                self.observation.extend(list(xyz))
            else:
                self.observation.extend([self.tx_act, self.ty_act, self.tz_act])

            # note, per-frame noise here
            # info of bottom height already included in tz
            if self.obs_noise:
                half_height_est = utils.perturb_scalar(
                    self.np_random, self.objs[self.top_id]["height"] / 2.0, 0.01,
                )
            else:
                half_height_est = self.objs[self.top_id]["height"] / 2.0
            self.observation.extend([half_height_est])
        else:
            xy = np.array([self.tx, self.ty])
            self.observation.extend(list(xy))
            if self.obs_noise:
                self.observation.extend(list(xy))
            else:
                self.observation.extend([self.tx_act, self.ty_act])

        # btm obj shape is not important.
        if self.objs[self.top_id]["shape"] == "box":
            shape_info = [1, -1, -1]
        elif self.objs[self.top_id]["shape"] == "cylinder":
            shape_info = [-1, 1, -1]
        elif self.objs[self.top_id]["shape"] == "sphere":
            shape_info = [-1, -1, 1]
        else:
            assert False
        self.observation.extend(shape_info)

        if self.use_gt_6d:
            self.vision_counter += 1
            # model both delayed and low-freq vision input
            # every vision_skip steps, update cur 6D
            # but feed policy with last-time updated 6D
            if self.vision_counter % self.vision_skip == 0:
                self.update_delayed_obj_6d(self.top_id)
                if self.btm_id:
                    self.update_delayed_obj_6d(self.btm_id)

            self.observation.extend(
                self.obj6DtoObs_UpVec(
                    self.objs[self.top_id]["last_position"],
                    self.objs[self.top_id]["last_orientation"],
                    is_sph=(self.objs[self.top_id]["shape"] == "sphere"),
                )
            )
            if self.btm_id:
                self.observation.extend(
                    self.obj6DtoObs_UpVec(
                        self.objs[self.btm_id]["last_position"],
                        self.objs[self.btm_id]["last_orientation"],
                    )
                )
            else:
                self.observation.extend(
                    self.obj6DtoObs_UpVec([0.0, 0, 0], [0.0, 0, 0, 1])
                )
            # print(self.objs[self.btm_id]['last_position'])
            # clPos_act, _ = p.getBasePositionAndOrientation(self.btm_id)
            # print("bact", clPos_act)

        if self.with_stack_place_bit:
            if self.place_floor:
                self.observation.extend([1.0])
            else:
                self.observation.extend([-1.0])

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = (
                self.np_random
            )  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s


if __name__ == "__main__":
    env = InmoovShadowHandPlaceEnvV9()
    p.setPhysicsEngineParameter(numSolverIterations=200)
    env.seed(303)
    for _ in range(20):
        env.reset()
        env.robot.tar_fin_q = env.robot.get_q_dq(env.robot.fin_actdofs)[0]
        for test_t in range(300):
            thumb_pose = [
                -0.84771132,
                0.60768666,
                -0.13419822,
                0.52214954,
                0.25141182,
            ]
            open_up_q = np.array([0.0, 0.0, 0.0] * 4 + thumb_pose)
            devi = open_up_q - env.robot.get_q_dq(env.robot.fin_actdofs)[0]
            if test_t < 200:
                env.robot.apply_action(np.array([0.0] * 7 + list(devi / 150.0)))
            p.stepSimulation()
            # input("press enter")
            if env.renders:
                time.sleep(env._timeStep * 2.0)
        print(env.robot.get_q_dq(env.robot.fin_actdofs))
    # input("press enter")
    p.disconnect()
