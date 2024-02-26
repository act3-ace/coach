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

import os
import time
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml

from utilities.PZWrapper import PettingZooEnv
from coach import COACHEnvironment
from env import COACH_PettingZoo

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
import supersuit as ss

from pettingzoo.sisl import waterworld_v4

import torch 
CORES = 6
torch.set_num_threads(CORES)
torch.set_num_interop_threads(CORES)

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

pymunk_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("pymunk")]
for log_handler in pymunk_loggers:
    log_handler.setLevel(logging.INFO)

# SB code adapted from https://pettingzoo.farama.org/tutorials/sb3/waterworld/

## We're going to train two policies, one that assumes a densely poisonous env
## and one that requires sparse exploration. 
# %%
def train(
    env_fn, 
    steps: int = 10_000, 
    seed: int | None = 0, 
    num_vec_envs=1, 
    num_cpus=1,
    learning_rate=1e-3,
    batch_size=256,
    model_dir=""
):
    os.makedirs(model_dir,exist_ok=True)

    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn()
    env.reset(seed=seed)
    print(env.possible_agents)

    logger.info(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tensorboard_log=os.path.join("./tensorboard_log/", model_dir.split("/")[-1])
    )
    
    eval_env = env_fn()
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    eval_callback = EvalCallback(eval_env, verbose=1, eval_freq=500)

    model.learn(total_timesteps=steps, callback=eval_callback)
    
    model_name = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)

    logger.info("Model has been saved.")
    logger.info(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    return model_path + ".zip"

def eval(
    env_fn, 
    model_path,
    num_games: int = 10, 
    render = False, 
    **env_kwargs
):
    # Evaluate a trained agent vs a random agent
    env = env_fn(**env_kwargs)

    logger.info(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render={render})"
    )

    model = PPO.load(model_path)

    cum_rewards = []

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in tqdm(range(num_games)):
        if render:
            env.coa_env.start_rendering()

        obs, info = env.reset(seed=i)
        
        rewards = {agent: 0 for agent in env.possible_agents}
        running = True
        j = 0
        while running:
            j += 1
            act = {"director": model.predict(obs["director"], deterministic=False)[0]}

            logger.debug("Obs From Director: %s", obs["director"]) # DEBUG
            logger.debug("Action From Director: %s", act) # DEBUG

            obs, reward, term, trunc, info = env.step(act)
            for a in env.agents:
                rewards[a] += reward[a]

            if all([a or b for a,b in zip(term.values(), trunc.values())]):
                for a in env.agents:
                    cum_rewards.append(rewards[a])
                
                running = False

        if render:
            frames = env.coa_env.state.trajectory.frames
            imgs = [Image.fromarray(frame) for frame in frames]
            # duration is the number of milliseconds between frames; this is 40 frames per second
            model_dir = model_path.split(".")[0]
            imgs[0].save(model_dir + f"_eval_run_{i}.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)
        
    env.close()

    avg_reward = np.mean(cum_rewards)
    std_reward = np.std(cum_rewards)

    with open(model_path + ".txt","a") as f:
        f.writelines(f"\n\n{env_kwargs}\n")
        f.writelines(f"\t Avg reward: {avg_reward}, std: {std_reward}\n")
        f.writelines(f"\t Rewards: {cum_rewards}\n")

    logger.info(f"\t Avg reward: {avg_reward}, std: {std_reward}")
    return avg_reward


def get_env():
    return PettingZooEnv(PZGame=waterworld_v4)

if __name__ == "__main__":
    MODEL_PATH = ""
    DENSE_PATH = min(glob.iglob(os.path.join(MODEL_PATH,'dense/*.zip')), key=os.path.getctime)
    EXPLORE_PATH = min(glob.iglob(os.path.join(MODEL_PATH,'explore/*.zip')), key=os.path.getctime)

    with open(os.path.join("Experiment", "train_director.yaml"), "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for agent, param in params["COACH_params"]["Agents"].items():
        param["params"]["policy_paths"]["dense"] = DENSE_PATH
        param["params"]["policy_paths"]["explore"] = EXPLORE_PATH

    env = COACH_PettingZoo(env_creator=get_env, COACHEnvClass=COACHEnvironment)

    def get_env_pz():
        env = COACH_PettingZoo(env_creator=get_env, COACHEnvClass=COACHEnvironment)
        env.augment(params)
        env.reset()
        return env

    model_path = train(
        get_env_pz,
        steps=100, 
        seed=0, 
        model_dir="director"
        )

    model_path = min(glob.iglob(os.path.join(MODEL_PATH,'director/*.zip')), key=os.path.getctime)
    rew = eval(
        get_env_pz,
        model_path=model_path,
        num_games=1,
        render=True
        )

    with open(os.path.join("Experiment", "dash_app_template.yaml"), 'r') as f:
        with open(os.path.join("Experiment", "dash_app.yaml"), 'w') as g:
            for line in f.readlines():
                line = line.replace("<DENSE_PATH>", DENSE_PATH)
                line = line.replace("<EXPLORE_PATH>", EXPLORE_PATH)
                line = line.replace("<DIRECTOR_PATH>", model_path)
                g.write(line)
    
# %%
