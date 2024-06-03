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

from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv
import gymnasium as gym
from typing import Any
import numpy as np
import argparse
from gymnasium.utils import EzPickle

from coach import (
    SeededParameterGenerator,
    BaseParameterGenerator,
)

from utilities.planning import COA

from coach import COACHEnvironment, CommunicationSchedule

import logging
logger = logging.getLogger(__name__)


def get_env_class(args: argparse.Namespace):
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env objects
    """
    return COACH_PettingZoo

def env(**kwargs):
    return COACH_PettingZoo(**kwargs)

def parallel_env(**kwargs):
    return COACH_PettingZoo(**kwargs)

class COACH_PettingZoo(gym.Wrapper, ParallelEnv):
    def __init__(self, env_creator, COACHEnvClass, AgentsModule, SBCompatabilityMode=True):
        # EzPickle.__init__(self)
        
        self.env_creator = env_creator
        self.COACHEnvClass = COACHEnvClass
        self.AgentsModule = AgentsModule
        self.fake_render_mode = "rgb_array"
        self.SBCompatabilityMode = SBCompatabilityMode

    def _setup(self, params):
        self.current_params = params
        self.COACH_params = COACH_params = params["COACH_params"]
        self.env_params = env_params = params["env_params"]

        # Set up parameter generator
        if COACH_params["stochastic"]:
            self.parameter_generator = SeededParameterGenerator(23423, env_params)
        else:
            self.parameter_generator = BaseParameterGenerator(23423, env_params)

        # Set up comm schedule
        if "FIXED_STEPS_PER_COM" in COACH_params.keys():
            schedule_param = COACH_params["FIXED_STEPS_PER_COM"]
            self.comm_schedule = CommunicationSchedule.repeating(**schedule_param)
        else:  
            logger.info("Communication Schedule Is Non-repeating")
            self.comm_schedule = CommunicationSchedule(length=0)
        
        self.ACTION_PADDING = COACH_params.get("ACTION_PADDING", 0)
        self.MIN_NEXT_ACTION_TIME = COACH_params.get("MIN_NEXT_ACTION_TIME", 1)
        self.MAX_NEXT_ACTION_TIME = COACH_params.get("MAX_NEXT_ACTION_TIME", np.inf)

        # Setup Agents
        agent_dict = dict()

        if "Agents" in self.COACH_params.keys():
            for role, agent in self.COACH_params["Agents"].items():
                logger.info(f"############# {role} {agent}")
                agent_class = self.AgentsModule.Agents[agent["class_name"]]
                agent_dict[role] = agent_class(role, **agent.get("params", dict()))

        # Create COA Env
        self.coa_env = self.COACHEnvClass(
            env_creator=self.env_creator,
            parameter_generator=self.parameter_generator,
            agents=agent_dict,
            fill_random=False,
            comm_schedule=self.comm_schedule,
            seed=COACH_params["seed"],
        )


        self.coa_env.augment(self.parameter_generator)
        self.coa_env.reset()

        self.stochastic = COACH_params["stochastic"]

        self.players = list(self.coa_env.agents.keys())
        self.player_actions = self.coa_env.action_spaces

        self.possible_agents = ["director"]
        self.agents = ["director"]

        # Action spaces
        self.player_actions = dict()
        all_lows = []
        all_highs = []
        self.action_indexs = []
        idx = 0
        for role in self.players: # Preserve order. Dicts should do this now but just to be safe
            # Process actions
            actions = self.coa_env.action_spaces[role]
            lows = []
            highs = []
            self.action_indexs
            for label, action in actions.items():
                # Need to add an inital entry for the start time
                lows.append(np.concatenate([[self.MIN_NEXT_ACTION_TIME], np.array(action.low).reshape(-1)]))
                highs.append(np.concatenate([[self.MAX_NEXT_ACTION_TIME], np.array(action.high).reshape(-1)]))

                L = len(np.array(action.low).reshape(-1)) + 1
                self.action_indexs.append((role, label, idx, idx + L))
                idx = idx + L

            lows = np.concatenate(lows)
            highs = np.concatenate(highs)
            all_lows.append(lows)
            all_highs.append(highs)
            self.player_actions[role] = Box(low = lows, high = highs)

        self.action_spaces = {"director": Box(
                low = np.concatenate(all_lows), 
                high = np.concatenate(all_highs)
            )
        }
        
        # Observation spaces
        self.player_observations = dict()
        all_lows = []
        all_highs = []
        for role in self.players: # Preserve order. Dicts should do this now but just to be safe
            # Process actions
            observations = self.coa_env.observation_spaces[role]
            low = np.array(observations.low).reshape(-1)
            high = np.array(observations.high).reshape(-1)

            all_lows.append(low)
            all_highs.append(high)
            self.player_observations[role] = observations

        self.observation_spaces = {"director": Box(
                low = np.concatenate(all_lows), 
                high = np.concatenate(all_highs)
            )
        }

############################################################
# Standard PettingZoo Interface Functions
############################################################
    def observation_space(self, role):
        return self.observation_spaces[role]

    def action_space(self, role):
        return self.action_spaces[role]

    def __getstate__(self):
        state = {
            "env_creator": self.env_creator,
            "COACHEnvClass": self.COACHEnvClass,
            "AgentsModule": self.AgentsModule,
            "fake_render_mode": self.fake_render_mode,
            "SBCompatabilityMode": self.SBCompatabilityMode,
            "current_params": self.current_params,
        }

        return state

    def __setstate__(self, newstate):
        for k,v in newstate.items():
            self.__dict__[k] = v
        
        self.augment(self.current_params)

    def reset(self, seed=0, options=None):
        # if self.SBCompatabilityMode:
        #     self.augment({"COACH_params":dict()})

        self.coa_env.reset()
        self.steps = 0
        self.cummulative_rew = np.zeros(len(self.possible_agents))

        self.coas = dict()
        for role in self.players:
            self.coas[role] = COA()

        return (
            {"director": self._process_observations(self.coa_env.last())},
            {},
        )

    def augment(self, params):
        self._setup(params)

    def seed(self, seed):
        self.parameter_generator.setseed(seed)

    def render(self, components=None, ao="rewards"):
        return self.coa_env.plot_trajectory_component(
            self.coa_env.state.trajectory,
            components = components,
            ao = ao
        )

    def step(self, action, render=False):
        self._process_actions(action)
        logger.debug("Director Step: COA: %s", self.coas)                           
        last_returns, step_end = self.coa_env.step(self.coas)
        logger.debug("Time: %s, Step End: %s", self.coa_env.state.current_t, step_end)

        # Process terminations and tructions
        term = False
        trunc = False

        if step_end["term_or_trunc"]:
            if all(list(last_returns[2].values())):
                # Unless everyone terminates, somebody must have trucated. 
                term = True
            else:
                trunc = True

        # Process reward
        reward_til_now = sum(self.coa_env.state.cummulative_rews.values())
        step_reward = reward_til_now - self.cummulative_rew
        self.cummulative_rew = reward_til_now

        return (
            {"director": self._process_observations(last_returns)},
            {"director": step_reward.reshape(-1)[0]},  ## Reward
            {"director": term},  ## Term
            {"director": trunc},  ## Trunc
            {"director": {}},  ## Info
        )
    
############################################################
# Converstion between actions and COAs
############################################################

    def _process_actions(self, action):
        action = action["director"]
        
        coas = {role:[] for role in self.players}

        logger.debug("Processing Action to COA")
        for role, label, i0, i1 in self.action_indexs:
            logger.debug("%s, %s, %s, %s, %s, %s", role, label, i0, i1, np.floor(action[i0]) + self.coa_env.state.current_t, action[i0+1:i1])
            event = {
                "start": np.floor(action[i0]) + self.coa_env.state.current_t,
                "label": label,
                "parameters": action[i0+1:i1],
                "role": role,
            }
            coas[role].append(event)
        
        for role in self.players:
            self.coas[role].add_events_from_dict(coas[role])

    def _process_observations(self, last_returns):
        # When working with a specific env you almost certanly want to change this
        # as there may be a ton of redudenent information in the combined observation
        # space
        obs = np.concatenate([last_returns[0][role].reshape(-1) for role in self.players])
        return obs.reshape(-1)

############################################################
# Wrapper Functions
############################################################

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""

        if name == "coa_env":
            if "coa_env" not in self.__dict__.keys():
                self.__dict__["coa_env"] = None
            return self.__dict__["coa_env"]

        if name in self.__dict__:
            return self.__dict__[name]

        if name == "unwrapped":
            return self.coa_env.env

        if name == "parallel_env":
            return self

        return getattr(self.coa_env.env, name)
    
############################################################
# DASH Viewer Functions
############################################################
        
    def set_fake_render_mode(self, fake_render_mode):
        self.fake_render_mode = fake_render_mode

# %%
if __name__ == "__main__":
    params = {
        "COACH_params":{
            "stochastic": True,
            # "FIXED_STEPS_PER_COM": {
            #     "checkin_frequency": 10
            # },
            "ACTION_PADDING": 0,
            "MIN_NEXT_ACTION_TIME": 1,
            "MAX_NEXT_ACTION_TIME": 10,
            "Agents": {
                "pursuer_0": {
                    "class_name": "DefaultActor", 
                    "params": {"max_action_len":6}
                },
                "pursuer_1": {
                    "class_name": "DefaultActor", 
                    "params": {"max_action_len":5}
                },
            },
            "seed": 453413,
        },

        "env_params": {"n_pursuers":2}
    }

    env = COACH_PettingZoo(env_creator=PettingZooEnv, COACHEnvClass=COACHEnvironment)
    
    env.augment(params)
    env.reset()

    print("#"*20, "COA Gym Information", "#"*20)
    print("Players:", env.players)
    print("Observation Space:", env.observation_spaces["director"].shape)
    print("Action Space:", env.action_spaces["director"].shape)
    print("Sample action:", env.action_spaces["director"].sample())
    
    for i in range(50):
        print("Turn", i)
        obs,rew,term,trunc,info = env.step({"director": env.action_spaces["director"].sample()})
        if all([a or b for a,b in zip(term.values(), trunc.values())]):
            print("Environment has terminated.")
            break
    
    env.render(ao="actions")
# %%
