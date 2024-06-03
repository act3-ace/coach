# %%
# Copyright (c) 2024 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from gymnasium.utils import EzPickle

from gymnasium import spaces
from pettingzoo.sisl import waterworld_v4
from pettingzoo.utils.conversions import parallel_wrapper_fn

"""
pygame - used for rendering
pymunk - used for object handeling and physics

This function converts coordinates in pymunk into pygame coordinates.

The coordinate system in pygame is:
            (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
                |       |                           │
                |       |                           │
(0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↓ y
The coordinate system in pymunk is:
(0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↑ y
                |       |                           │
                |       |                           │
            (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
"""

class WW_Waypoint(BaseWrapper, EzPickle):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        BaseWrapper.__init__(self, *args, **kwargs)
        ## Need to add a waypoint. This means we need to adjust the observation space

        # waterworld_base: LN 284 - Sample random coordinate (x, y) with x, y ∈ [0, pixel_scale]
        self._base_env = self.env.unwrapped.env
        self._unwrapped = self.env.unwrapped
        obs_space = self._base_env.observation_space[0]
        self.scale = self._base_env.pixel_scale

        low = np.concatenate([obs_space.low, [0,0]])
        high = np.concatenate([obs_space.low, [self.scale,self.scale]])
        
        obs_space = spaces.Box(
            low=low,
            high=high,
            shape=(len(low),), 
        )

        agents = ["pursuer_" + str(r) for r in range(self._base_env.num_agents)]
        self.observation_spaces = {agt:obs_space for agt in agents}
        
        self.WAYPOINT_PROXIMITY = self.scale/10
        self.WAYPOINT_ARRIVE_REWARD = 20
        self.WAYPOINT_APPROCH_REWARD_SCALE = 10

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        # Generate new Waypoints
        self.waypoints = np.random.random(size=[self.env.num_agents,2]) * self.scale

    ## Need to take this from the file
    def step(self, action):
        agent = self.agent_selection
        super().step(action)

        # Reward for how close to the waypoint
        dist_to_wp = np.linalg.norm(self.relative_waypoint(agent))
        r = self.agent_name_mapping[agent]
        rwd = self.WAYPOINT_APPROCH_REWARD_SCALE/dist_to_wp

        # Check if we're at the waypoint
        if dist_to_wp < self.WAYPOINT_PROXIMITY:
            rwd += self.WAYPOINT_ARRIVE_REWARD
            self.waypoints[r,:] = np.random.random(size=2) * self.scale

        # print(self._unwrapped.rewards)
        if agent in self._unwrapped.rewards:
            self._unwrapped.rewards[agent] += rwd
            self._unwrapped._cumulative_rewards[agent] += rwd

    
    def relative_waypoint(self, agent):
        idx = self.agent_name_mapping[agent]
        pursuer = self._base_env.pursuers[idx]
        wp = self.waypoints[idx]
        rel_wp = wp - pursuer.body.position

        return rel_wp

    def observe(self, agent):
        obs = self.env.observe(agent)
        rel_wp = self.relative_waypoint(agent)
        return np.concatenate([obs, rel_wp])

    def render(self, mode="rgb_array"):
        img = super().render()
        img_scale = img.shape[0] 
        a = 3
        for wpt in self.waypoints:
            pt = img_scale * wpt/self.scale
            pt[1] = img_scale - pt[1]
            img[int(pt[1])-a:int(pt[1])+a, int(pt[0])-a:+int(pt[0])+a,:] = 0

            r = self.WAYPOINT_PROXIMITY
            N = 60
            for i in range(N):
                y = int(np.min([pt[1] + r * np.cos(i*2*np.pi/N), self.scale-1]))
                x = int(np.min([pt[0] + r * np.sin(i*2*np.pi/N), self.scale-1]))
                img[y,x] = 0

        return img

def env(**kwargs):
    env = waterworld_v4.env(render_mode="rgb_array", max_cycles=100)
    env = WW_Waypoint(env)
    return env

def parallel_env(**kwargs):
    return parallel_wrapper_fn(env)(**kwargs)

# %%
if __name__ == "__main__":
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()


# %%
