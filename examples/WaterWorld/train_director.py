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

# %%

#%%
from __future__ import annotations
import os
import time
import sys 
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, "../../")

from utilities.PZWrapper import PettingZooEnv
from utilities.sb3 import trainPPO

from coach import COACHEnvironment
import examples.WaterWorld.director_env as DirectorEnv

import examples.WaterWorld.agents as WaypointAgentModule
import pettingzoo.sisl.waterworld_v4 as TrainingEnv

from stable_baselines3 import PPO
import supersuit as ss

import torch 
CORES = 2
torch.set_num_threads(CORES)
torch.set_num_interop_threads(CORES)

# COACH has slightly different rendering considerations than other env's 
# since each "step" in the coach env may be multiple steps in sim. This 
# uses COACH's underlying rendering framework where rendering can be
# turned on or off and saved into the trajectory. 

def eval_WaterWorld(
    env_fn, 
    model_path,
    num_games: int = 1, 
    render = True, 
    **env_kwargs
):  
    # Evaluate a trained agent vs a random agent
    env = env_fn(**env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render={render})"
    )

    model = PPO.load(model_path)

    cum_rewards = []

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
            new_frames = []

            for i in range(1, len(frames)):
                info = env.coa_env.state.trajectory.agent_info[i]
                waypoints = [p["waypoint"] for p in info.values()]
                scale = env.coa_env.env.unwrapped.env.pixel_scale
                frame = add_waypoint(frames[i], waypoints, scale)
                new_frames.append(frame)
             
            imgs = [Image.fromarray(frame) for frame in new_frames]

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

# Function to add a waypoint to the frame rendered by the waterworld env. 
def add_waypoint(img, waypoints, scale, a=8, r=75):
    img_scale = img.shape[0] 
    for wpt in waypoints:
        if wpt is not None:
            # print(wpt)
            pt = img_scale * wpt/scale
            pt[1] = img_scale - pt[1]
            img[int(pt[1])-a:int(pt[1])+a, int(pt[0])-a:+int(pt[0])+a,:] = 0

            N = 60
            for i in range(N):
                y = int(np.min([pt[1] + r * np.cos(i*2*np.pi/N), scale-1]))
                x = int(np.min([pt[0] + r * np.sin(i*2*np.pi/N), scale-1]))
                img[y,x] = 0

    return img

def get_env():
    return PettingZooEnv(PZGame=TrainingEnv)

if __name__ == "__main__":
    # Load model in for Waypointing
    WAYPOINTER_PATH = "/root/coach/examples/WaterWorld/test/waterworld_v4_20240521-153057/model.zip"
    
    # We want to use a numpy model so we can properly multithread
    tmp_model = PPO.load(WAYPOINTER_PATH)
    tmp_np_model = WaypointAgentModule.SB_PPO_Standard_MLP(tmp_model).save(WAYPOINTER_PATH[:-3] + ".json")

    del tmp_model, tmp_np_model

    with open(os.path.join("Experiment", "train_director.yaml"), "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tmp_env = get_env()
    tmp_env.augment(params["env_params"])
    tmp_env.reset()        
    
    # Generate this automaticaly
    for agent in tmp_env.possible_agents:
        params["COACH_params"]["Agents"][agent] = {
            "class_name": "SB_PPOWaypointActor",
            "params": {"policy_path": WAYPOINTER_PATH}
        }

    # Setup the Coach Environment. Requires a creator for hte underlying environemnt
    # and the module for the file
    def get_env_pz():
        env = DirectorEnv.env(
            env_creator=get_env, 
            COACHEnvClass=COACHEnvironment,
            AgentsModule=WaypointAgentModule
        )
        env.augment(params)
        env.reset()
        return env

    model_path = trainPPO(
        get_env_pz,
        steps=100, 
        seed=0, 
        model_dir="director"
        )

    eval_WaterWorld(
        model_path=model_path,
        env_fn=get_env_pz
    )
    
# %%
