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

import sys

from utilities.timelines import TimelineEvent
from utilities.planning import COA
from numpy.random import PCG64DXSM, Generator
import numpy as np
import copy
import logging
from gymnasium.spaces import Box

logger = logging.getLogger(__name__)

from collections import OrderedDict
from stable_baselines3 import PPO
import sys, inspect

##################################################
##
## Interfaces
##
##################################################

class ActionBox(Box):
    def __init__(self, low=[], high=[], shape=[], default=[], description=None):
        if len(low)>0:
            super().__init__(low=low, high=high, shape=shape)
        else:
            self.low = []
            self.high = []

        self.description = description

        if len(default) == len(low):
            self.default = default
        elif len(default) == 0:
            default = []
            for l, h in zip(low, high):
                # If the bounds are finite, take the average.
                if np.isfinite(l) and np.isfinite(h):
                    default.append((l+h)/2)
                # If one is infinte, we're going to assume the other is the default value
                elif np.isinf(l) and np.isfinite(h):
                    default.append(h)
                elif np.isfinite(l) and np.isinf(h):
                    default.append(l)
                ## Otherwise, assume 0
                else:
                    default.append(0)
            
            self.default = default
        else:
            raise Exception("Action Box has default values of length different than box.")

    def __repr__(self) -> str:
        return f"{self.description}: low {self.low}, high {self.high}"

    def items(self):
        actions = []
        for i in range(len(self.low)):
            actions.append(
                ActionBox(
                    low = np.array([self.low[i]]), 
                    high = np.array([self.high[i]]), 
                    shape = (1,), 
                    description = self.description[i]
                )
            )

        return actions

    def equals(self, other):
        # All defualt interfaces are the same
        if not (type(other) is type(self)):
            return False
        if not all([a==b for a,b in zip(self.low, other.low)]):
            return False
        if not all([a==b for a,b in zip(self.high, other.high)]):
            return False
        if not all([a==b for a,b in zip(self.shape, other.shape)]):
            return False
        if not all([a==b for a,b in zip(self.default, other.default)]):
            return False

        return True


class TrivialInterface:
    def __init__(self, role, env):
        self.env_observation_space = copy.deepcopy(env.observation_space(role))
        self.env_action_space = copy.deepcopy(env.action_space(role))

        self.action_space = self.env_action_space
        self.observation_space = self.env_observation_space

        self.action_dictionary = {"default": ActionBox()}
        
        self.name = self.__class__.__name__

    def action_names(self):
        return list(self.action_dictionary.keys())

    def get_action_descriptions(self):
        # Return human readable description of the sceion parameters.
        return self.action_description

    def get_action_dictionary(self):
        # Return the dctionary of possible actions with their parameter
        # spaces
        return self.action_dictionary

    # We may want to verify and error check that an interface matches an expected
    # interface, espeaclly if the interface takes parameters. 
    def equals(self, other):
        # All defualt interfaces are the same
        if type(other) is type(self):
            return True
        return False


class DefaultInterface(TrivialInterface):
    def __init__(self, role, env, max_action_len=None):
        super().__init__(role, env)

        if not max_action_len:
            max_action_len = np.inf

        # The burn is "legnth" + "action space"
        burn = ActionBox(
            low=np.concatenate([(0,), self.env_action_space.low]),
            high=np.concatenate([(max_action_len,), self.env_action_space.high]),
            shape=(1 + np.prod(self.env_action_space.shape),),
            description = ["Length"] + ["Unknown"] * len(self.env_action_space.low)
        )

        self.action_dictionary = {"burn": burn}

    def equals(self, other):
        # All defualt interfaces are the same
        if type(other) is type(self):
            return self.action_dictionary["burn"].equals(self.action_dictionary["burn"])

        return False


class RandomActionInterface(TrivialInterface):
    def __init__(self, role, env):
        super().__init__(role, env)

        self.action_dictionary = {"random": ActionBox()}


class SBPolicyInterface(TrivialInterface):
    def __init__(self, role, env, n_policies, max_action_len):
        super().__init__(role, env)
        
        one_hot_low = [0]*n_policies
        one_hot_high = [1]*n_policies
        desc = ["Action Length"] + [f"policy_{i}" for i in range(n_policies)]

        policies = ActionBox(
            low=np.array([0] + one_hot_low),
            high=np.array([max_action_len] + one_hot_high),
            shape=(len(one_hot_low) + 1,),
            description=desc
        )

        self.action_dictionary = {"Policies": policies}

    def equals(self, other):
        if type(other) is type(self):
            return self.action_dictionary["Policies"].equals(other.action_dictionary["Policies"])
        return False
    

##################################################
##
## Actors
##
##################################################


class BasicActor:
    InterfaceType = TrivialInterface

    def __init__(self, role, reference_interface:TrivialInterface = None):
        self.role = role
        self.reference_interface = reference_interface

    def action_names(self):
        return list(self.interface.action_dictionary.keys())

    def get_action_dictionary(self):
        # Return the dctionary of possible actions with their parameter
        # spaces
        return self.interface.action_dictionary

    def get_action_descriptions(self):
        # Return human readable description of the sceion parameters.
        return self.interface.action_description

    def get_action(self, obs, t, mean_mode=False):
        return self.none_action, {"acting": False, "coa_done": True}
    
    def process_observations(self, obs, t):
        # Agents may communicate different information than what they observe
        return obs

    def update_coa(self, coa):
        self.coa.add_timeline_to(coa, allow_dup_labels=False)

    def reset(self, env):
        self.env = env
        self.interface = BasicActor.InterfaceType(self.role, env)
        if self.reference_interface is not None:
            if not self.interface.equals(self.reference_interface):
                raise Exception("Actor interface does not match reference interface.")

        self.coa = COA()

        self.none_action = np.zeros(self.interface.action_space.shape)


class DefaultActor(BasicActor):
    InterfaceType = DefaultInterface

    def __init__(self, role, max_action_len=None):
        super().__init__(role)
        self.max_action_len = max_action_len

    def __str__(self):
        return f"DefaultActor: {self.role}"

    def __repr__(self):
        return self.__str__()

    def get_action(self, obs, t, mean_mode=False):
        if t in self.coa.timeline.keys():
            e = self.coa.get(time = t, label = "burn")
            self.burn_time = np.floor(e.parameters[0])
            self.current_burn = np.array(e.parameters[1:]).reshape(
                self.interface.action_space.shape
            )

        time_stamps = list(self.coa.timeline.keys())
        if len(time_stamps) == 0:
            coa_done = True
        else:
            last_t = max(time_stamps)
            last_duration = self.coa.get(time = last_t, label= "burn").parameters[0]

            if t >= last_t + last_duration:
                coa_done = True
            else:
                coa_done = False

        if self.burn_time > 0:
            self.burn_time -= 1
            return self.current_burn, {"acting": True, "coa_done": coa_done}
        else:
            return self.none_action, {"acting": False, "coa_done": coa_done}

    def reset(self, env=None):
        super().reset(env)

        self.interface = DefaultActor.InterfaceType(self.role, env, max_action_len=self.max_action_len)

        self.current_burn = None
        self.burn_time = 0


class RandomActor(BasicActor):
    InterfaceType = RandomActionInterface

    def __init__(self, role, seed=23143):
        super().__init__(role)
        self.seed = seed

    def __str__(self):
        return f"RandomActor: {self.role}"

    def __repr__(self):
        return self.__str__()

    def get_action(self, action, parameters):
        r = self.np_random.uniform(
            low=self.interface.action_space.low, high=self.interface.action_space.high
        )

        return r, {"acting": True, "coa_done": False}

    def reset(self, env):
        super().reset(env)

        self.interface = RandomActor.InterfaceType(self.role, env)
        self.np_random = Generator(PCG64DXSM(seed=self.seed))


class SB_PPOPoliciesActor(BasicActor):
    InterfaceType = SBPolicyInterface

    def __init__(self, role, policy_paths: dict, max_action_len: int, interface:SBPolicyInterface=None):
        super().__init__(role) 

        self.max_action_len = max_action_len

        self.policies = dict()
        for policy, model_path in policy_paths.items():
            # For some reason SB doesn't want to zip on the end
            if model_path[-3:] == "zip":
                model_path = model_path.split(".")[0]
            self.policies[policy] = PPO.load(model_path)

        self.policy_names = list(self.policies.keys())

    def __str__(self):
        return f"PPOPoliciesActor: {self.role}, Policies: {self.policies.keys()}"

    def get_action(self, obs, t, mean_mode=False):
        bad_command = False
        acting = False
        coa_done = False

        if t in self.coa.timeline.keys():
            ## Start new policy
            e = self.coa.get(time = t, label = "Policies")
            self.time_remaining = e.parameters[0]*self.max_action_len
            logger.debug("Agent: start new policy %s", e) # DEBUG
            logger.debug("policy: %s, timeleft: %s", self.policy_names[int(np.argmax(e.parameters[1:]))], self.time_remaining)  # DEBUG

            self.current_policy = self.policy_names[int(np.argmax(e.parameters[1:]))]

        # self.next_waypoint_abs = np.array([7.08763559,  -25.61246231, -167.51736098])
        if (self.current_policy is not None) and (self.time_remaining > 0):
            acting = True
            action, _ = self.policies[self.current_policy].predict(obs, deterministic=True)
            self.time_remaining += -1
        else:
            action = copy.copy(self.none_action)

        if self.time_remaining == 0:
            coa_done = True

        return action, {
            "acting": acting,
            "coa_done": coa_done,
            "bad_command": bad_command,
        }

    def reset(self, env):
        super().reset(env)
        self.time_remaining = -1
        self.current_policy = None

        self.interface = SB_PPOPoliciesActor.InterfaceType(
            self.role, 
            env, 
            n_policies=len(self.policy_names), 
            max_action_len=self.max_action_len
        )

        if self.reference_interface is not None:
            if not self.interface.equals(self.reference_interface):
                raise Exception("Actor interface does not match reference interface.")


classes = [
    cls_obj
    for cls_name, cls_obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(cls_obj) and cls_obj.__module__ == __name__
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
