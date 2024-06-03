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
import copy
import logging
from gymnasium.spaces import Box
logger = logging.getLogger(__name__)
from numpy.random import PCG64DXSM, Generator
from agents import *
import agents as agts_super
from examples.MAInspection.Environments.env import hills_frame
from utilities.sb3 import SB_PPO_Standard_MLP

class WaypointInterface(TrivialInterface):
    def __init__(self, role, env):
        super().__init__(role, env)
    
        desc = ["rel frame x", "rel frame y", "rel frame z"]

        policies = ActionBox(
            low=-np.ones(3)*env._MAX_OUTER_PERIMETER.value,
            high=np.ones(3)*env._MAX_OUTER_PERIMETER.value,
            shape=(3,),
            description=desc
        )

        self.action_dictionary = {"Waypoint": policies}

    def equals(self, other):
        if type(other) is type(self):
            return self.action_dictionary["Waypoint"].equals(other.action_dictionary["Waypoint"])
        return False


class SB_PPOWaypointActor(BasicActor):
    InterfaceType = WaypointInterface 
    def __init__(self, role, policy_path: str, interface:WaypointInterface=None):
        super().__init__(role) 

        if policy_path[-3:] == "zip":
            policy_path = policy_path[:-4] + '.json'

        tmp = SB_PPO_Standard_MLP()
        tmp.load(policy_path)
        self.policy = tmp
        self.next_waypoint = None

    def __str__(self):
        return f"SB_PPOWaypointActor: {self.role}"
        
    def get_action(self, obs, t, mean_mode=False):
        bad_command = False
        acting = False
        coa_done = False
        action = copy.copy(self.none_action)

        if t in self.coa.timeline.keys():
            ## Start new waypoint
            e = self.coa.get(time = t, label = "Waypoint")
            # self.time_remaining = e.parameters
            self.next_waypoint = e.parameters
            logger.debug("Agent: start new waypoint %s", e) # DEBUG

        
        if self.next_waypoint is not None:
            i = self.env.possible_agents.index(self.role)

            pos = self.env.orb[self.role].r.to(self.env._OU_DIS).value
            pos_c = self.env.orb["chief"].r.to(self.env._OU_DIS).value

            # if self.env.OBSERVATION_FRAME == "Hills":
            #     frame = hills_frame(self.env.orb[self.role])

            # else:
            #     frame = self.env.ori[i] 
            frame = self.env._local_frame("Chief Hills", self.role)

            rel_waypt = frame.T @ (self.next_waypoint + pos_c - pos).reshape(-1,1)

            agent_obs = np.concatenate([obs[:6*self.env.num_deputies],rel_waypt.reshape(-1)])
            if np.linalg.norm(rel_waypt) < self.env._WAYPOINT_ARRIVAL_PROX.value:
                acting = False
                coa_done = True
                self.next_waypoint = None
            else:
                acting = True
                action = self.policy.predict(agent_obs)

        return action, {
            "acting": acting,
            "coa_done": coa_done,
            "bad_command": bad_command,
        }

    def reset(self, env):
        super().reset(env)
        self.time_remaining = -1
        self.current_policy = None

        self.interface = SB_PPOWaypointActor.InterfaceType(
            self.role, 
            env
        )

        self.env = env

        if self.reference_interface is not None:
            if not self.interface.equals(self.reference_interface):
                raise Exception("Actor interface does not match reference interface.")


classes = [
    cls_obj
    for cls_name, cls_obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(cls_obj) and ((cls_obj.__module__ == __name__) or (cls_obj.__module__ == agts_super.__name__))
]

Interfaces = {}
for c in classes:
    # BasicActor
    if issubclass(c, TrivialInterface):
        Interfaces[c] = []

Agents = {}
for c in classes:
    # BasicActor
    if issubclass(c, BasicActor):
        Interfaces[c.InterfaceType].append(c)
        Agents[c.__name__] = c