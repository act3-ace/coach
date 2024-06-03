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

# The following code is modified from Farama-Foundation/PettingZoo
# (https://github.com/Farama-Foundation/PettingZoo)
# under the MIT License.

"""PettingZoo Wrapper"""

import argparse  # for type hinting
import logging
from collections.abc import Callable  # for type hinting
from typing import Any, Union, cast  # for type hinting

from gymnasium.wrappers import FlattenObservation
from pettingzoo.classic import connect_four_v3 as PZGame  # type: ignore[import]
from pettingzoo.sisl import waterworld_v4 as PZGame

logging.getLogger("pettingzoo.utils.env_logger").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import copy

import numpy as np
from gymnasium.spaces import Box, Discrete, flatten, flatten_space, unflatten
from numpy.typing import NDArray
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.env import AECEnv, ParallelEnv
from gymnasium.utils import EzPickle

from .PZParams import PettingZooEnvParams


################################################################################
## Auxiliary Functions
################################################################################
def softmax(x):
    exp = np.exp(x)
    return exp / cast(float, np.exp(x).sum())


@unflatten.register(Discrete)
def _unflatten_discrete(
    space: Discrete, x: Union[NDArray[np.int64], NDArray[float]]
) -> np.int64:
    nonzero = np.nonzero(x)
    if len(nonzero[0]) == 0:
        raise ValueError(
            f"{x} is not a valid one-hot encoded vector and can not be unflattened to space {space}. "
            "Not all valid samples in a flattened space can be unflattened."
        )

    act = np.argmax(x)

    return space.start + act


#######################################################
## Factories
#######################################################
def get_env_class(args: argparse.Namespace) -> Callable[..., Any]:
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
    return PettingZooEnv


## This is what you want to edit if you add wrappers, or change
## things about the env parameters.
def env_creator(**kwargs):
    return PZGame.env(**kwargs)


################################################################################
## Main Class
################################################################################
class PettingZooEnv():
    def __init__(self, PZGame) -> None:
        """ """
        # EzPickle.__init__(self, PZGame)
        def env_creator(**kwargs):
            return PZGame.parallel_env(**kwargs)
        
        self.env_creator = env_creator
        # If you have wrappers define a function to set it up properly
        self.standard_params = {"render_mode": "rgb_array"}
        self.current_params = None

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "env":
            if "env" not in self.__dict__.keys():
                self.__dict__["env"] = None
            return self.__dict__["env"]

        return getattr(self.env, name)

    def unflatten_or_none(self, action_space, action):
        if action is not None:
            return unflatten(action_space, action)
        else:
            return None

    def step(self, act):
        act = {agt: self.unflatten_or_none(self.env.action_space(agt), v) for agt, v in act.items()}

        # logger.info(f"action: {str(act)}")
        if self.parallel:
            return self.env.step(act)
        else:
            self.env.step(act[self.env.agent_selection])

        obs = {
            agt: flatten(self.env.observation_space(agt), self.env.observe(agt))
            for agt in self.env.possible_agents
        }

        # logger.info(f"action: {str(obs)}, {self.env.terminations}, {self.env.truncations}")
        return (
            obs,
            self.env.rewards,
            self.env.terminations,
            self.env.truncations,
            self.env.infos,
        )

    def reset(self):
        if self.parallel:
            return self.env.reset()
        
        self.env.reset()
        # print(obs)
        obs = {
            agt: self.env.observation_space(agt)
            for agt in self.env.possible_agents
        }
        # return obs, info
        return obs, self.env.info

    def augment(self, params):
        self.current_params = copy.copy(self.standard_params)

        for k, v in params.items():
            self.current_params[k] = v

        self.env = self.env_creator(**self.current_params)
        if isinstance(self.env, ParallelEnv):
            self.parallel = True
        else:
            self.parallel = False

        self.env.reset()

        self.action_spaces = dict()
        self.observation_spaces = dict()

        self.action_type = dict()
        for agt in self.env.possible_agents:
            self.action_spaces[agt] = flatten_space(self.env.action_space(agt))
            self.observation_spaces[agt] = flatten_space(
                self.env.observation_space(agt)
            )
            self.action_type[agt] = type(self.env.action_space(agt))

    def seed(self, seed):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def render(self, *args, **kwargs):
        if kwargs.get("close", False):
            self.env.close()
            return

        return self.env.render()
