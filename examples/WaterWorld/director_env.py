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

from env import COACH_PettingZoo
from gymnasium.spaces import Box
import numpy as np
import copy

def env(**kwargs):
    return WW_Coach(**kwargs)

def parallel_env(**kwargs):
    return WW_Coach(**kwargs)
    

class WW_Coach(COACH_PettingZoo):
    def _setup(self, params):
        super()._setup(params)

        self.base_env = self.coa_env.env.unwrapped.env
        self.objects = []
        self.objects += self.base_env.evaders
        self.objects += self.base_env.pursuers

        N = len(self.objects)

        self.observation_spaces = {"director": Box(
                low = np.zeros(2*N), 
                high = np.ones(2*N)
            )}

        self.real_action_spaces = copy.deepcopy(self.action_spaces)

        N = len(self.real_action_spaces["director"].low)
        self.action_spaces = {"director": Box(
                low = np.zeros(N), 
                high = np.ones(N), 
            )}

    def reset(self, seed=None, options=None):
        objects = []
        objects += self.coa_env.env.unwrapped.env.evaders
        objects += self.coa_env.env.unwrapped.env.pursuers

        obs, info = super().reset(seed, options)
        obs = np.array([copy.deepcopy(p.body.position) for p in objects]).reshape(-1)/self.base_env.pixel_scale

        return {"director":obs}, info

    def step(self, action):
        aspace = self.real_action_spaces["director"]
        action = aspace.low + (aspace.high - aspace.low) * (action["director"])

        objects = []
        objects += self.coa_env.env.unwrapped.env.evaders
        objects += self.coa_env.env.unwrapped.env.pursuers

        obs, rwd, term, trunc, info = super().step({"director":action})
        obs = np.array([copy.deepcopy(p.body.position) for p in objects]).reshape(-1)/self.base_env.pixel_scale

        return {"director": obs}, rwd, term, trunc, info