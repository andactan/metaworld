import numpy as np

# from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_door import SawyerDoorEnv, _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.kuka.sawyer_door_kuka import SawyerDoorKukaEnv, _assert_task_is_set

class SawyerDoorCloseKukaEnv(SawyerDoorKukaEnv):
    def __init__(self):
        super().__init__()

        # modify init configs
        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0.95, 0.1, 0.1], dtype=np.float32), # todo: might need calibration
            'hand_init_pos': np.array([0.6, 0, 0.2], dtype=np.float32), # todo: might need calibration
        }

        # modify goal position
        self.goal = np.array([0.8, -0.2, 0.15]) # todo: might need calibration
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        self.random_init = True

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy() + np.array([-0.15, -0.15, 0.05]) # todo: might need calibration
            self._target_pos = goal_pos

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos

        # keep the door open after resetting initial positions
        self._set_obj_xyz(-1.5708)
        self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('handle')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPullDist + 1000*2

        return self._get_obs()

    def compute_reward(self, actions, obs):
        del actions
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._target_pos

        pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]

