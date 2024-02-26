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
sys.path.insert(0, "../../")

import numpy as np
import copy
import logging
from gymnasium.spaces import Box

logger = logging.getLogger(__name__)
from collections import OrderedDict
import matplotlib.pyplot as plt
from numpy.random import PCG64DXSM, Generator
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import examples.MAInspection.Environments.MAIagents as agents
import io
from PIL import Image

# %matplotlib inline
from coach import (
    COACHEnvironment, 
    COA, 
    State, 
    Trajectory, 
    BaseParameterGenerator, 
    SeededParameterGenerator,
    CommunicationSchedule,
    Timeline
)

# %%
def plot_orbits(orbits):
    trace = go.Scatter3d(
        x=orbits[:, 0],
        y=orbits[:, 1],
        z=orbits[:, 2],
        mode="markers",
        marker=dict(color=orbits[:, 3:], size=5),
    )

    return trace

    # ax.scatter(orbits[:,0], orbits[:,1], orbits[:,2], c = orbits[:,3:], alpha = alpha)


def plot_points(p_array, c):
    # c = np.zeros([seen.shape[0], 3])
    c = mpl.colors.to_rgba_array(c)
    tmp = np.zeros([p_array.shape[0], 7])
    tmp[:, :3] = p_array
    tmp[:, 3:] = c

    t_seen = go.Scatter3d(
        x=tmp[:, 0],
        y=tmp[:, 1],
        z=tmp[:, 2],
        mode="markers",
        marker=dict(color=tmp[:, 3:], size=5),
    )

    return t_seen


def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


class MAI_Trajectory(Trajectory):
    def __init__(self, env, initial_return, intial_frame=None):
        self.pos = []
        self.ori = []
        self.pts = []
        self.unobserved_points = []
        self.cumm_rewards = []

        super().__init__(env, initial_return, intial_frame=intial_frame)

    def add(self, env, action, step_return, agent_info, frame=None):
        super().add(env, action, step_return, agent_info, frame=frame)
        frame = env.hills_frame(env.orb["chief"])
        
        # Here: would be frame.T because we're mapping from absolute coords to 
        # chief frame, so right mul is the same
        # for orientation, recall that self.ori[k, axis, :] is the k'th 
        # agent's axis, with axis 0 being the heading. 

        self.pos.append(copy.copy(env.pos) @ frame)
        self.ori.append(copy.copy(env.ori) @ frame)
        self.pts.append(copy.copy(env.pts) @ frame)
        self.unobserved_points.append(copy.copy(env.unobserved_points))
        self.cumm_rewards.append(copy.copy(env.cum_rewards))


class MAI_COACH(COACHEnvironment):
    agent_selection = agents

    def __init__(
        self,
        env_creator: callable,
        parameter_generator:BaseParameterGenerator,
        agents=dict(),
        fill_random=True,
        comm_schedule:Timeline=None,
        seed=6234235,
        TrajectoryClass = MAI_Trajectory,
    ):
        super().__init__(env_creator,
            parameter_generator,
            agents=agents,
            fill_random=fill_random,
            comm_schedule=comm_schedule,
            seed=seed,
            TrajectoryClass = TrajectoryClass
        )

    def plot_trajectory_component(
        self, 
        traj, 
        coas=None, 
        env=None, 
        plan=None,
        render_mode="plotly", 
        simulate_data=False
    ):
        if plan:
            coas = plan.coas

        players = traj.players
        cmap = plt.get_cmap("tab10")
        event_cmap = plt.get_cmap("Set1")
        player_to_color = {p: i / 10 for i, p in enumerate(players)}

        game_len = len(traj)
        num_points = env.num_points
        chief_perim = env._CHIEF_PERIMETER
        
        # 3 dim: pos, 4 dim: color
        xs = np.zeros([game_len, len(players), 7])
        xs[:,:,:3] = np.array([pos.to(env._OU_DIS).value for pos in traj.pos])

        # Plot Cone Information
        CONE_FREQ = 5
        CONE_OPATICY = 0.5

        n_cones = int(np.ceil(game_len/CONE_FREQ))
        cones = np.zeros([n_cones*len(players), 10])

        # Stack is going to concatenate long the first axis, combine all 
        # players info into one stack
        cones[:, :3] = np.vstack(xs[::CONE_FREQ,:,:3])
        cones[:, 3:6] = np.vstack(np.array(traj.ori)[::CONE_FREQ,:,:])[:,0,:]
        pcolors = [cmap(player_to_color[p]) for p in players]
        cones[:, 6:] = np.repeat(np.array(pcolors),n_cones, axis=0)
        cones[:, -1] = CONE_OPATICY

        # Seperate into drifts and burns
        for p_idx, p in enumerate(players):
            c = cmap(player_to_color[p])

            if p in coas.keys():
                coa = coas[p]
            else:
                coa = COA()

            event_time_left = 0

            for i in range(game_len):
                if i in coa.timeline.keys():
                    for event in coa.timeline[i]:
                        # event = coa.timeline[i][action]
                        event_time_left = event.parameters[0]
                        print(event.label)
                        c = event_cmap(event.id)
                        print(c)

                if event_time_left <= 0:
                    c = cmap(player_to_color[p])

                xs[i,p_idx,3:] = c
                event_time_left -= 1

        ## Get Seen Points
        pts = traj.pts[0]
        u_set = traj.unobserved_points[0]

        unseen = np.zeros(pts.shape[0], dtype=bool)
        unseen[list(u_set)] = True

        seen = pts[~unseen]
        unseen = pts[unseen]

        traces = []
        traces.append(plot_orbits(np.vstack(xs)))

        if unseen.shape[0] > 0:
            traces.append(plot_points(unseen, "#000000"))

        if seen.shape[0] > 0:
            traces.append(plot_points(seen, "#FFFFFF"))

        # # Draw Chief Sphere
        (x_pns_surface, y_pns_surface, z_pns_suraface) = ms(0, 0, 0, chief_perim)
        traces.append(
            go.Surface(
                x=x_pns_surface,
                y=y_pns_surface,
                z=z_pns_suraface,
                opacity=0.5,
                showscale=False,
            )
        )

        # View Cones:
        traces.append(
            go.Cone(
                x=cones[:, 0],
                y=cones[:, 1],
                z=cones[:, 2],
                u=cones[:, 3],
                v=cones[:, 4],
                w=cones[:, 5],
                opacity=0.1,
                showscale=False,
                showlegend=False,
                sizemode="absolute",
                sizeref=10,
                anchor="tip",
            )
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene_aspectmode="cube",
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    nticks=4,
                    range=[-300, 300],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[-300, 300],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-300, 300],
                ),
            ),
            width=700,
            margin={"l": 40, "b": 40, "t": 10, "r": 0},
            hovermode="closest",
        )

        if render_mode == "plotly":
            return fig

        if render_mode == "rgb_array":
            return plotly_fig2array(fig)


#################################################################
## Example Usage
#################################################################

if __name__ == "__main__":
    from examples.MAInspection.env import MultInspect

    ## Environemntal Parameters
    env_params = {"_OBS_REWARD":.01, "num_deputies": 2}

    # Actor Parameters
    COACH_params = {
        "Agents": {
            "player_0": {
                "class_name":"SB_PPOWaypointActor",
                "params": {"policy_path": "/root/coach/examples/MAInspection/waypointer/MAInspect_20240205-234515.zip"}    # If the agent has setup parameters
                },
            "player_1": {
                "class_name":"SB_PPOWaypointActor",
                "params": {"policy_path": "/root/coach/examples/MAInspection/waypointer/MAInspect_20240205-234515.zip"}    # If the agent has setup parameters
                },
        }
    }

    # Create actors from parameters
    agent_dict = dict()
    if "Agents" in COACH_params.keys():
        for role, agent in COACH_params["Agents"].items():
            agent_class = MAI_COACH.agent_selection.Agents[agent["class_name"]]
            agent_dict[role] = agent_class(role, **agent.get("params", dict()))

    # Example communication scheudle. 
    comms = CommunicationSchedule.repeating(checkin_frequency=10)

    # Wrap parameters in parameter generator
    parameter_generator = SeededParameterGenerator(23423, env_params)

    def env_creator():
        return MultInspect(**env_params)

    env = MAI_COACH(
        env_creator = MultInspect,
        parameter_generator = parameter_generator,
        comm_schedule = comms,
        fill_random = True,
        agents = agent_dict
    )

    for agt in env.agents.values():
        print(agt.interface)

    for i in range(2):
        env.step()
    traj = env.state.trajectory
    coas = {role: agt.coa for role, agt in env.agents.items()}
    fig = env.plot_trajectory_component(traj, coas, env=env.env)
    fig.show()
# %%
