from .base_kalman import BaseKalman
import numpy as np 

class NSAKalman(BaseKalman):

    def __init__(self, ):

        state_dim = 8  # [x, y, a, h, vx, vy, va, vh]
        observation_dim = 4 

        F = np.eye(state_dim, state_dim)
        '''
        [1, 0, 0, 0, 1, 0, 0]
        [0, 1, 0, 0, 0, 1, 0]
        ...
        '''
        for i in range(state_dim // 2):
            F[i, i + state_dim // 2] = 1

        H = np.eye(state_dim // 2, state_dim)
    
        super().__init__(state_dim=state_dim, 
                       observation_dim=observation_dim, 
                       F=F, 
                       H=H)
        
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initialize(self, observation):
        """ init x, P, Q, R
        
        Args:
            observation: x-y-a-h format
        """
        # init x, P, Q, R

        mean_pos = observation
        mean_vel = np.zeros_like(observation)
        self.kf.x = np.r_[mean_pos, mean_vel]  # x_{0, 0}

        std = [
            2 * self._std_weight_position * observation[3],  # related to h
            2 * self._std_weight_position * observation[3], 
            1e-2, 
            2 * self._std_weight_position * observation[3], 
            10 * self._std_weight_velocity * observation[3], 
            10 * self._std_weight_velocity * observation[3], 
            1e-5, 
            10 * self._std_weight_velocity * observation[3], 
        ]       

        self.kf.P = np.diag(np.square(std))  # P_{0, 0}

    def predict(self, is_activated=True):
        """ predict step

        x_{n + 1, n} = F * x_{n, n} 
        P_{n + 1, n} = F * P_{n, n} * F^T + Q

        """
        std_pos = [
            self._std_weight_position * self.kf.x[3],
            self._std_weight_position * self.kf.x[3],
            1e-2,
            self._std_weight_position * self.kf.x[3]]
        std_vel = [
            self._std_weight_velocity * self.kf.x[3],
            self._std_weight_velocity * self.kf.x[3],
            1e-5,
            self._std_weight_velocity * self.kf.x[3]]
        
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.kf.predict(Q=Q)
        
    def update(self, z, score):
        """ update step
        
        Args:
            z: observation x-y-a-h format
            score: the detection score/confidence required by NSA kalman

        K_n = P_{n, n - 1} * H^T * (H P_{n, n - 1} H^T + R)^{-1}
        x_{n, n} = x_{n, n - 1} + K_n * (z - H * x_{n, n - 1})
        P_{n, n} = (I - K_n * H) P_{n, n - 1} (I - K_n * H)^T + K_n R_n

        """

        std = [
            self._std_weight_position * self.kf.x[3],
            self._std_weight_position * self.kf.x[3],
            1e-1,
            self._std_weight_position * self.kf.x[3]]
        
        # NSA 
        std = [(1. - score) * x for x in std]
        
        R = np.diag(np.square(std))

        self.kf.update(z=z, R=R)
