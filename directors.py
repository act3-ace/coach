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
from stable_baselines3 import PPO

import copy

import logging
logger = logging.getLogger(__name__)


class SB3_PPO_Director:
    def __init__(self, env_creator, COACHEnvClass, params, model_path):
        self.env_creator = env_creator
        self.COACHEnvClass = COACHEnvClass
        self.params = params

        self.env = COACH_PettingZoo(env_creator=env_creator, COACHEnvClass=COACHEnvClass)
        self.env.augment(params)

        if model_path.endswith(".zip"):
            model_path = model_path[:-4]

        self.policy = PPO.load(model_path)
        
    def generate_coas(self, params=None):
        if not params:
            params = self.params

        self.env.augment(params)

        obs, info = self.env.reset()
        running = True

        while running:
            act = {"director": self.policy.predict(obs["director"], deterministic=False)[0]}

            # logger.debug("Obs From Director: %s", obs["director"]) # DEBUG
            # logger.debug("Action From Director: %s", act) # DEBUG

            obs, reward, term, trunc, info = self.env.step(act)

            if all([a or b for a,b in zip(term.values(), trunc.values())]):
                running = False

        coas = dict()
        for role, agent in self.env.coa_env.agents.items():
            coas[role] = agent.coa
        
        traj = copy.deepcopy(self.env.coa_env.state.trajectory)
        return coas, traj