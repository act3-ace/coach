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

import examples.MAInspection.Environments.env as TrainingEnv
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
from PIL import Image
import os
import time
import glob
from tqdm import tqdm
import numpy as np
import yaml
import copy

from utilities.sb3 import SB_PPO_Standard_MLP

import torch 
CORES = 7
torch.set_num_threads(CORES)
torch.set_num_interop_threads(CORES)

# Code adapted from https://pettingzoo.farama.org/tutorials/sb3/waterworld/

## We're going to train two policies, one that assumes a densely poisonous env
## and one that requires sparse exploration. 

def train_butterfly_supersuit(
    env_fn, 
    steps: int = 10_000, 
    seed: int | None = 0, 
    num_vec_envs=1, 
    num_cpus=CORES,
    learning_rate=1e-3,
    batch_size=256,
    model_dir="",
    name="",
    **env_kwargs
):  
    model_path = os.path.join(model_dir, name)
    os.makedirs(model_path,exist_ok=True)

    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tensorboard_log=os.path.join("./tensorboard_log/", name)
    )

    eval_env = env_fn(**env_kwargs)
    eval_env.reset(seed=seed)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")

    eval_callback = EvalCallback(eval_env, verbose=1, eval_freq=1000)

    weight_name = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    weight_path = os.path.join(model_path, weight_name)
    os.makedirs(weight_path,exist_ok=True)

    model_args = {
        "name": name,
        "load_class": "SB_PPOWaypointerActor",
        "env_params": env_kwargs
    }

    with open(os.path.join(weight_path, "params.yaml"), "w") as f:
        yaml.dump(model_args, f, default_flow_style=False)

    eval_callback = EvalCallback(eval_env, verbose=1, eval_freq=1000)
    model.learn(total_timesteps=steps, callback=eval_callback)
    model.save(os.path.join(weight_path, "model"))

    tmp = SB_PPO_Standard_MLP(model)
    tmp.save(os.path.join(weight_path, "model.json"))
    del tmp
    
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    return os.path.join(weight_path, "model")


def eval(
    env_fn, 
    model_path,
    num_games: int = 5, 
    render_mode: str | None = "rgb_array", 
    render = True,
    **env_kwargs
):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    print(model_path)

    model = PPO.load(model_path)

    total_rewards = []

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in tqdm(range(num_games)):
        obs, infos = env.reset(seed=i)
        cumm_rewards = {agent: 0 for agent in env.possible_agents}
        running = True
        frames = []

        while running:
            acts = dict()
            for role in env.agents:
                acts[role] = model.predict(obs[role], deterministic=True)[0]

            obs, rewards, terminations, truncations, infos = env.step(acts)

            for role in env.agents:
                cumm_rewards[role] += rewards[role]

            if all([a or b for a,b in zip(terminations.values(), truncations.values())]):
                running = False
                total_rewards += list(cumm_rewards.values())
            
            if render:
                frames.append(env.render())

        if render:
            imgs = [Image.fromarray(frame) for frame in frames]
            # duration is the number of milliseconds between frames; this is 40 frames per second
            model_dir = model_path.split(".")[0]
            imgs[0].save(model_dir + f"_eval_run_{i}.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)
            
    env.close()

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    with open(model_path + ".txt","a") as f:
        f.writelines(f"\n\n{env_kwargs}\n")
        f.writelines(f"\t Avg reward: {avg_reward}, std: {std_reward}\n")
        f.writelines(f"\t Rewards: {total_rewards}\n")

    print(f"\t Avg reward: {avg_reward}, std: {std_reward}")
    return avg_reward


if __name__ == "__main__":

    with open("Experiment/train_agents.yaml", "r") as stream:
        try:
            experiments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    STEPS = 2000

    # Train Waypoint Agent
    model_paths = dict()
    for name, experiment in experiments.items():
        model_paths[name] = train_butterfly_supersuit(
            env_fn=TrainingEnv.parallel_env, 
            steps=STEPS,
            model_dir=name,
            **experiment["env_params"]
            )

    for name, model_path in model_paths.items():#, model_path_explore]:
        eval(env_fn=TrainingEnv.parallel_env,
            model_path=model_path,
            **experiments[name]["env_params"]
            )