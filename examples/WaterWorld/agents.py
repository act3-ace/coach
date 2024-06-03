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

from agents import *
import agents as agts_super
from utilities.sb3 import SB_PPO_Standard_MLP
import copy

class WaypointInterface(TrivialInterface):
    def __init__(self, role, env):
        super().__init__(role, env)

        waypoint = ActionBox(
            low=np.zeros(2),
            high=np.ones(2),
            shape=(2,),
            description=["rel frame x", "rel frame y"]
        )

        self.action_dictionary = {"Waypoint": waypoint}

    def equals(self, other):
        if type(other) is type(self):
            return self.action_dictionary["Waypoint"].equals(other.action_dictionary["Waypoint"])
        return False


class SB_PPOWaypointActor(BasicActor):
    InterfaceType = WaypointInterface 
    def __init__(self, role, policy_path: str, interface:WaypointInterface=None):
        super().__init__(role) 

        self.policy = SB_PPO_Standard_MLP()
        self.policy.load(policy_path[:-4] + ".json")

        self.next_waypoint = None



    def relative_waypoint(self, wp, agent):
        idx = self.env.agent_name_mapping[agent]
        pursuer = self.base_env.pursuers[idx]
        rel_wp = wp - pursuer.body.position

        return rel_wp

    def get_action(self, obs, t, mean_mode=False):
        bad_command = False
        acting = False
        coa_done = False
        action = copy.copy(self.none_action)

        if t in self.coa.timeline.keys():
            ## Start new waypoint
            e = self.coa.get(time = t, label = "Waypoint")
            self.next_waypoint = e.parameters*self.scale
            # print("Next Waypoint: ", self.next_waypoint)
            logger.debug("Agent: start new waypoint %s", e) # DEBUG

        
        if self.next_waypoint is not None:
            rel_waypt = self.relative_waypoint(self.next_waypoint, self.role)
            agent_obs = np.concatenate([obs,np.array([rel_waypt]).reshape(-1)])

            if np.linalg.norm(rel_waypt) < self.scale/10:
                # print("Got to waypoint")
                acting = False
                coa_done = True
                self.next_waypoint = None
            else:
                acting = True
                action = self.policy.predict(agent_obs)
                action = self.truncate_to_action_box(action)
        else:
            agent_obs = np.concatenate([obs,np.zeros(2)])
            action = self.policy.predict(agent_obs)
            # action = np.zeros(2)
            acting = False
        
        # print({"acting": acting,
        #     "coa_done": coa_done,
        #     "bad_command": bad_command,
        #     "waypoint": copy.deepcopy(self.next_waypoint),
        #     })

        return action, {
            "acting": acting,
            "coa_done": coa_done,
            "bad_command": bad_command,
            "waypoint": copy.deepcopy(self.next_waypoint),
        }


    def reset(self, env):
        super().reset(env)
        self.interface = SB_PPOWaypointActor.InterfaceType(
            self.role, 
            env
        )

        self.env = env.unwrapped
        self.base_env = self.env.env
        self.scale = self.base_env.pixel_scale
        self.action_space = self.env.action_space(self.role)

        if self.reference_interface is not None:
            if not self.interface.equals(self.reference_interface):
                raise Exception("Actor interface does not match reference interface.")

    def truncate_to_action_box(self, data):
        # print(data.shape, self.low.shape)
        data = np.max(np.stack([data.reshape(self.action_space.shape), self.action_space.low]), axis=0)
        data = np.min(np.stack([data, self.action_space.high]), axis=0)
        return data



# np_model = SB_PPO_Standard_MLP(model)
# np_model.save("test.json")

# %%

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