import gym
import pandas as pd
import numpy as np
import torch

class NetworkTrafficEnv(gym.Env):
    """Custom Environment that follows gym interface for network traffic analysis"""
    metadata = {'render.modes': ['human']}
    count_true = []
    count_false = []

    def __init__(self, df: pd.DataFrame):
        super(NetworkTrafficEnv, self).__init__()
        """
        Initialize the environment according to OpenAI GYM.
        This environment provides network traffic data for training and evaluating reinforcement learning agents.
        """
        # Action space indicates 49 possible intrusions, you can custom this according to your own action space or classes.
        self.action_space = gym.spaces.Discrete(49)
        
        # Observation space since features are normalized between 0 and 1,
        # Excluding the last column since it is 'Label'.
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(df.shape[1] - 1,), dtype=np.float32)
        
        # DataFrame containing network traffic data.
        self.df = df
        
        # Index of the current row
        self.current_row_index = 0

    def step(self, action: int):
        """
        Execute one time step within the environment.
        """
        # Print Row index at specific intervals
        if self.current_row_index % 20000 == 0:
            print(self.current_row_index)
            
        # Retrieve the Label for the current state.
        correct_action = self.df.iloc[self.current_row_index]['Label']
        
        # Reward System
        reward = 10 if action == correct_action else -5

        # Count correct actions against incorrect actions
        if action == correct_action:
            if isinstance(action, torch.Tensor):
                self.count_true.append(action.item())
            else:
                self.count_true.append(action)
        else:
            if isinstance(action, torch.Tensor):
                self.count_false.append(action.item())
            else:
                self.count_false.append(action)

        # Observation for the next state
        observation = self.df.iloc[self.current_row_index][:-1].values
        self.current_row_index += 1
        
        # Check if we're done
        done = self.current_row_index >= len(self.df)

        return observation, reward, done, {}, self.count_true, self.count_false

    def reset(self):
        """
        Reset the state of the environment to get an initial state.
        """
        self.current_row_index = 0
        # Clear action counts
        self.count_true.clear()
        self.count_false.clear()
        # Return the initial observation
        return self.df.iloc[0][:-1].values

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen (Not applicable in this environment).
        """
        if mode == 'human':
            print(f"Current row index: {self.current_row_index}")

# Usage example:
# df = pd.read_csv('network_traffic_data.csv')
# env = NetworkTrafficEnv(df)
# observation = env.reset()
# while True:
#     action = env.action_space.sample()  # Sample random action
#     observation, reward, done, info, count_true, count_false = env.step(action)
#     env.render()
#     if done:
#         break
# env.reset()
