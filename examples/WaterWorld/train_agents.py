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

from __future__ import annotations
import sys
sys.path.insert(0, "../../")

import examples.WaterWorld.training_env as TrainingEnv
from utilities.sb3 import trainPPO, evalPPO

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

import supersuit as ss

from PIL import Image
import os
import time
import yaml

import torch 
CORES = 7
torch.set_num_threads(CORES)
torch.set_num_interop_threads(CORES)

## We're going to train a policy that has an agent go towards a waypoint while
## getting food and avoiding posiion. 

if __name__ == "__main__":
    STEPS = 2000

    # Train Waypoint Agent
    model_path = trainPPO(
        env_fn=TrainingEnv.parallel_env, 
        steps=STEPS,
        model_dir="waypoint",
        eval_args=
            {
                "log_path": "./waypoint_logs/",
                "best_model_save_path": "./waypoint_logs/"
            }
        )

    evalPPO(env_fn=TrainingEnv.parallel_env,
        model_path=model_path,
        )