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

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

import supersuit as ss

def trainPPO(
    env_fn, 
    steps: int = 10_000, 
    seed: int = 0, 
    num_vec_envs=2, 
    num_cpus=2,
    learning_rate=1e-3,
    batch_size=256,
    model_dir="",
    start_from=None,
    eval_args=dict(),
    model_args=dict()
):
    os.makedirs(model_dir,exist_ok=True)

    # Setup vector enviroments for training using supersuit 
    env = env_fn()
    env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    print(f"Starting training on {str(env.unwrapped.metadata['name'])}.")
    
    # Setup vector enviroments for evaluation using supersuit 
    eval_env = env_fn()
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    eval_arg_dict = {
        "verbose": 1,
        "eval_freq": 500,
        "log_path": "./logs/",
        "best_model_save_path": "./logs/",
        "deterministic": True,
        "render": False
    }

    for k,v in eval_args.items():
        eval_arg_dict[k] = v

    eval_callback = EvalCallback(
        eval_env, 
        **eval_arg_dict
    )

    # Setup Model
    if start_from is not None:
        # Load the model

        model_arg_dict = {
            "tensorboard_log": os.path.join("./tensorboard_log/", model_dir.split("/")[-1]),
        }

        for k,v in model_args.items():
            model_arg_dict[k] = v

        if start_from.endswith(".zip"):
            start_from = start_from[:-4]

        model = PPO.load(
            "/root/coach/examples/WaterWorld/director/waterworld_v4_20240530-013634",
            env=env,
            **model_arg_dict
        )

    else:
        # Create new model

        model_arg_dict = {
            "verbose": 3,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "tensorboard_log": os.path.join("./tensorboard_log/", model_dir.split("/")[-1]),
        }

        for k,v in model_args:
            model_arg_dict[k] = v

        model = PPO(
            MlpPolicy,
            env,
            **model_arg_dict
        )

    model.learn(total_timesteps=steps, callback=eval_callback)

    # Save Final Model
    model_name = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)

    with open(os.path.join(model_dir, "model_args.yaml"), "w") as f:
        yaml.dump(model_arg_dict, f, default_flow_style=False)

    print("Final model has been saved at %s", model_path)
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    return model_path + ".zip"

def evalPPO(
    env_fn, 
    model_path,
    num_games: int = 5, 
    render_mode: str = "rgb_array", 
    render = True,
    **env_kwargs
):
    # Evaluate a trained agent vs a random agent
    env = env_fn(**env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    print("Located at", model_path)

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


def evalPPO_COACH(
    env_fn, 
    model_path,
    num_games: int = 1, 
    render = True, 
    **env_kwargs
):  
    # COACH has slightly different rendering considerations than other env's 
    # since each "step" in the coach env may be multiple steps in sim. This 
    # uses COACH's underlying rendering framework where rendering can be
    # turned on or off and saved into the trajectory. 

    # Evaluate a trained agent vs a random agent
    env = env_fn(**env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render={render})"
    )

    model = PPO.load(model_path)

    cum_rewards = []

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in tqdm(range(num_games)):
        if render:
            env.coa_env.start_rendering(steps_per_frame=1)

        obs, info = env.reset(seed=i)
        
        rewards = {agent: 0 for agent in env.possible_agents}
        running = True
        j = 0
        while running:
            j += 1
            act = {"director": model.predict(obs["director"], deterministic=False)[0]}

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

    print(f"\t Avg reward: {avg_reward}, std: {std_reward}")
    return avg_reward



import torch
import json
from utilities.iotools import NumpyEncoder, NumpyDecoder
torch_to_idx = {
    torch.nn.modules.activation.Tanh: 0,
    "PASSTHROUGH":1
}
idx_to_np = {
    0: np.tanh,
    1: lambda x: x
}


class SB_PPO_Standard_MLP:
    def __init__(self, model=None):
        self.weights = []
        self.biases = []
        self.activations = []

        if model is not None:
            layers = [l for l in model.policy.mlp_extractor.policy_net]
            layers += [model.policy.action_net]

            for layer in layers:
                if type(layer) == torch.nn.modules.linear.Linear:
                    self.weights.append(layer._parameters['weight'].detach().numpy())
                    self.biases.append(layer._parameters['bias'].detach().numpy().reshape(-1,1))

                elif type(layer) in torch_to_idx:
                    self.activations.append(torch_to_idx[type(layer)])

            self.activations.append(torch_to_idx["PASSTHROUGH"])

    def predict(self, x):
        for i in range(len(self.weights)):
            x = self.weights[i] @ x.reshape(-1,1) + self.biases[i]
            x = idx_to_np[self.activations[i]](x)
        
        
        return(x)

    def save(self, filename):
        with open(filename, 'w') as jsonfile:
            json.dump(self.__dict__, jsonfile, cls=NumpyEncoder)

    def load(self, filename):
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile, cls=NumpyDecoder)
            for k,v in data.items():
                self.__dict__[k] = v