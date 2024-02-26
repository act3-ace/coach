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

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
import os
import time
import glob
from tqdm import tqdm
import numpy as np
import yaml
import copy

import torch 
CORES = 9
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
    env = env_fn.parallel_env(**env_kwargs)
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

    eval_env = env_fn.parallel_env(**env_kwargs)
    eval_env.reset(seed=seed)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")

    eval_callback = EvalCallback(eval_env, verbose=1, eval_freq=10000)

    weight_name = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    weight_path = os.path.join(model_path, weight_name)

    model_args = {
        "name": name,
        "load_class": "SB_PPOPoliciesActor",
        "env_params": env_kwargs
    }

    with open(weight_path + ".yaml", "w") as f:
        yaml.dump(model_args, f, default_flow_style=False)

    model.learn(total_timesteps=steps, callback=eval_callback)
    model.save(weight_path)


    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    return weight_path


def eval(
    env_fn, 
    model_path,
    num_games: int = 10, 
    render_mode: str | None = None, 
    **env_kwargs
):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    model = PPO.load(model_path)

    cum_rewards = []

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in tqdm(range(num_games)):
        env.reset(seed=i)
        rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                for a in env.agents:
                    cum_rewards.append(rewards[a])
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = np.mean(cum_rewards)
    std_reward = np.std(cum_rewards)

    with open(model_path + ".txt","a") as f:
        f.writelines(f"\n\n{env_kwargs}\n")
        f.writelines(f"\t Avg reward: {avg_reward}, std: {std_reward}\n")
        f.writelines(f"\t Rewards: {cum_rewards}\n")

    print(f"\t Avg reward: {avg_reward}, std: {std_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = waterworld_v4
    env_kwargs = {}

    STEPS = 1000

    ## Train Dense Eater
    ## Note: We want one eater here so that it doesn't learn to just sit there and hope for food.
    model_path_dense = train_butterfly_supersuit(
        env_fn=env_fn, 
        steps=STEPS, 
        seed=0, 
        n_pursuers=1,
        n_evaders=10,
        n_poisons=40,
        model_dir="",
        name="dense"
        )

    ## Train Explore Eater
    model_path_explore = train_butterfly_supersuit(
        env_fn=env_fn, 
        steps=STEPS, 
        seed=0, 
        n_pursuers=3,
        n_evaders=5,
        n_poisons=5,
        model_dir="",
        name="explore")

    for model_path in [model_path_dense, model_path_explore]:#, model_path_explore]:
        eval(env_fn=env_fn,
            model_path=model_path,
            n_pursuers=3,
            n_evaders=20,
            n_poisons=30,
            )

        eval(env_fn=env_fn,
            model_path=model_path,
            n_pursuers=3,
            n_evaders=5,
            n_poisons=5,
            )

        eval(env_fn=env_fn,
            model_path=model_path,
            n_pursuers=3,
            n_evaders=15,
            n_poisons=15,
            )