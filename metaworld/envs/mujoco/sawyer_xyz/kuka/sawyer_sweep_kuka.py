import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerSweepKukaEnv(SawyerXYZEnv):

    def __init__(self):

        # modify range configs
        init_puck_z = 0.1
        hand_low = (0.45, -0.45, 0.05)
        hand_high = (0.85, 0.45, 0.3)
        obj_low = (0.45, -0.1, 0.02)
        obj_high = (0.55, 0.1, 0.02)
        goal_low = (.65, -.44, -0.301) # ask this (z-value)
        goal_high = (.75, -.45, -0.299) # ask this (z-value)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        # modify init config
        self.init_config = {
            'obj_init_pos':np.array([0.5, 0., 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([.6, 0., .2]),
        }

        # modify goal position
        self.goal = np.array([0.75, 0., -0.3])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.init_puck_z = init_puck_z

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        # modify XML path
        return full_v1_path_for('sawyer_xyz/sawyer_sweep_kuka.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pushDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pushDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pushDist <= 0.05)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom').copy()

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((obj_pos[:2], [self.obj_init_pos[-1]]))
            goal_pos = obj_pos.copy()
            goal_pos[1] = -0.45 # east of the effector is -y (also modified)
            goal_pos[2] = -0.3
            self._target_pos = goal_pos

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.data.get_geom_xpos('objGeom')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPushDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pushGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)
        pushDistxy = np.linalg.norm(objPos[:-1] - pushGoal[:-1])
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        if objPos[-1] < self.obj_init_pos[-1] - 0.05:
            reachRew = 0
            pushDistxy = 0
            reachDist = 0

        def pushReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.reachCompleted:
                pushRew = 1000*(self.maxPushDist - pushDistxy) + c1*(np.exp(-(pushDistxy**2)/c2) + np.exp(-(pushDistxy**2)/c3))
                pushRew = max(pushRew,0)
                return pushRew
            else:
                return 0

        pushRew = pushReward()
        reward = reachRew + pushRew

        return [reward, reachDist, pushDistxy]
