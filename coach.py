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

from typing import Any
import numpy as np
import copy
import logging
logger = logging.getLogger(__name__)
import argparse

import matplotlib.pyplot as plt
from numpy.random import PCG64DXSM, Generator

import agents as agents
from agents import (
    BasicActor,
    DefaultActor,
    RandomActor
)

from utilities.timelines import Timeline, Timelines
from utilities.planning import (
    COA, 
    CommunicationSchedule, 
    BaseParameterGenerator, 
    SeededParameterGenerator, 
    State, 
    Trajectory,
    Telemetry,
) 

from matplotlib.backends.backend_agg import (
    FigureCanvasAgg as FigureCanvas,  # type: ignore[import]
)

# %%

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

    return COACHEnvironment

class COACHEnvironment:
    agent_selection = agents

    def __init__(
        self,
        env_creator: callable,
        parameter_generator:BaseParameterGenerator,
        agents=dict(),
        fill_random=True,
        comm_schedule:Timeline=None,
        seed=6234235,
        TrajectoryClass = Trajectory,
    ):
        self.env_creator = env_creator
        self.env = env_creator()
        self.parameter_generator = parameter_generator

        self.stored_agents = agents
        self.fill_random = fill_random
        self.seed = seed
        self.TrajectoryClass = TrajectoryClass

        if comm_schedule is None:
            comm_schedule = Timelines(labels=["blackout", "checkins"])
            comm_schedule.checkins.add_event(time=0, )
        self.comm_schedule = comm_schedule

        self.rendering=False

        self.reset()

        for k in agents.keys():
            if not k in self.env.possible_agents:
                raise Exception(
                    "Passed actor name is not in enviroments possible actors"
                )

    def __deepcopy__(self, memo):
        tmp_agents = {k: copy.deepcopy(a) for k, a in self.stored_agents.items()}

        tmp = self.__class__(
            self.env_creator,
            self.parameter_generator,
            tmp_agents,
            copy.deepcopy(self.fill_random),
            copy.deepcopy(self.comm_schedule),
            copy.deepcopy(self.seed),
        )

        tmp.setstate(self.state)
        return tmp

    def _fillagents(self, fill_random=None):
        if not fill_random:
            fill_random = self.fill_random

        self.action_spaces = {}
        self.observation_spaces = {}

        self.agents = dict()

        for agt in self.env.possible_agents:
            if agt in self.stored_agents.keys():
                self.agents[agt] = self.stored_agents[agt]
            else:
                if fill_random:
                    self.agents[agt] = RandomActor(agt)
                else:
                    self.agents[agt] = DefaultActor(agt)

            self.agents[agt].reset(self.env)
            self.action_spaces[agt] = self.agents[agt].interface.action_dictionary
            self.observation_spaces[agt] = self.agents[agt].interface.observation_space

    def possible_models(self):
        default = BasicActor, DefaultActor, RandomActor
        return {role: default for role in self.env.possible_agents}

    def set_coa(self, coas):
        for agt, coa in coas.items():
            self.agents[agt].update_coa(coa)

    def get_coa(self):
        return {role: agt.coa for role, agt in self.agents.items()}

    def simulate(
        self, 
        coas=None, 
        parameters=None, 
        agents=None,
        state=None,
        time_steps=None, 
        comm_steps=None, 
        render=False
    ):
        if comm_steps and time_steps:
            logger.info(
                "Only one of `time_steps` and `comm_steps` can be defined at a time"
            )
            return

        tmp_env = copy.deepcopy(self)
        if agents is not None:
            tmp_env.stored_agents = agents

        if render:
            tmp_env.start_rendering()

        # Set new env params:
        if parameters:
            tmp_env.augment(parameters)
        
        tmp_env.reset()
        tmp_agts = tmp_env.agents
        
        if state:
            if parameters:
                logger.info(
                    "Note: State parameters will overwrite passed parameters."
                )
            tmp_env.setstate(state)

        # Set new COA for agents
        if coas:
            for agt, coa in coas.items():
                tmp_agts[agt].update_coa(coa)

        # Run simulation, recording trajectory
        if time_steps:
            tmp_env.step_env(steps=time_steps)

        elif comm_steps:
            for _ in range(comm_steps):
                tmp_env.step()
        
        else:
            ended = False
            while not ended:
                tmp_env.step()
                terms = list(tmp_env.step_return[-1][2].values())
                truncs = list(tmp_env.step_return[-1][3].values())
                if all([a or b for a, b in zip(terms, truncs)]):
                    ended = True

        # Since trajectory starts at 0'th step
        trajectory = tmp_env.state.trajectory

        del tmp_env
        del tmp_agts

        return trajectory
    
    def start_rendering(self, steps_per_frame = 10):
        self.steps_per_frame = steps_per_frame
        self.rendering=True

    def stop_rendering(self):
        self.rendering=False

    def augment(self, parameters):
        self.parameter_generator = parameters

    def reset(self):
        seed, parameters = self.parameter_generator.sample()
        self.setup_env(seed, parameters)
        self.state.cummulative_rews = {agt: 0 for agt in self.env.possible_agents}

    def setup_env(self, seed, parameters):
        self.env.augment(parameters)
        self.env.seed(int(seed))
        obs, info = self.env.reset()
        self.step_return = [(obs, None, None, None, info)]

        # Fill agents based on reset env
        self._fillagents()

        ## Reset agents, get their initial conditions.
        self.agent_info = [
            {role: agt.reset(env=self.env) for role, agt in self.agents.items()}
        ]

        initial_frame = None
        if self.rendering:
            initial_frame = self.env.render()

        traj = self.TrajectoryClass(
            self.env, 
            initial_return=self.step_return[0], 
            intial_frame=initial_frame
        )

        self.state = State(parameters, seed, traj, current_t=0)
        self.alive = {agt:True for agt in self.agents.keys()}


    def setstate(self, state: State):
        self.setup_env(state.seed, state.parameters)
        
        for step in state.trajectory[1:]:
            step_return = self.env.step(step["action"])
            self.step_return.append(step_return)

            self.state.trajectory.add(
                env=self.env,
                action=step["action"],
                agent_info=step["agent_info"],
                step_return=step_return,
            )

        self.state.current_t = len(state.trajectory) - 1


    def last(self):
        return self.step_return[-1]

    def step(self, coas=None):
        ## This is the step through the coa
        if coas:
            self.set_coa(coas)

        next_comm, _ = self.comm_schedule.checkins.next_event(self.state.current_t)
        if next_comm:
            next_comm = next_comm - self.state.current_t

        logger.debug("current_t: %s, next_com: %s", self.state.current_t, next_comm)    # DEBUG
        self.step_env(steps=next_comm)

        return self.step_return[-1], self.step_end

    def step_env(self, steps=1, coas=None):
        self.step_end = {
            "coa_done": [],
            "term_or_trunc": False,
            "steps_reached": False
        }

        if coas:
            self.set_coa(coas)

        t = self.state.current_t-1
        running = True

        while running:
            t += 1
            logger.debug("Env Step: %s", t)  # DEBUG

            action = dict()
            agent_info = dict()

            for role, agt in self.agents.items():
                if self.alive[role]:
                    action[role], agent_info[role] = agt.get_action(
                        self.step_return[-1][0][role], t
                    )
                else:
                    action[role] = None
                    agent_info[role] = "not_alive"

            logger.debug("Env Action: %s", action)    # DEBUG

            obs, rwds, terms, truncs, info = self.env.step(action)

            frame = None
            if self.rendering:
                if t % self.steps_per_frame == 0:
                    frame = self.env.render()

            returns = (
                {role: self.agents[role].process_observations(o, t) for role, o in obs.items()},
                rwds, terms, truncs, info
            )

            for role, rwd in returns[1].items():
                self.state.cummulative_rews[role] += rwd

            self.agent_info.append(agent_info)
            self.step_return.append(returns)
            self.state.trajectory.add(
                env=self.env, 
                action=action, 
                step_return=returns, 
                agent_info=agent_info,
                frame=frame)

            self.alive = {agt: not (returns[2][agt] or returns[3][agt]) for agt in self.agents.keys()}

            break_for_new_COA = False

            if self.comm_schedule.ALLOW_AGENT_BREAK:
                for role in self.agents.keys():
                    if self.comm_schedule.blackouts.get(t, role) is not None:
                        if self.agent_info[-1][role]["coa_done"]:
                            self.step_end["coa_done"].append(role)
                            break_for_new_COA = True
                            running = False
                    
            if not any(self.alive.values()):
                self.step_end["term_or_trunc"] =  True
                running = False

            if steps:
                if t >= (self.state.current_t + steps):
                    running = False
            
        if steps:
            if t == self.state.current_t + steps - 1:
                # We reached the end, even if other things would have terminated it
                self.step_end["steps_reached"] = True

        logger.debug("Env step end: %s", self.step_end)  # DEBUG
        self.state.current_t = t + 1

#####################################
# COA Level Interface:
#####################################

    def get_traj_from_coas(self, coas, params=None, from_state=None, render=False):
        traj = self.simulate(
            coas,
            parameters=params,
            state=from_state,
            render=render
            )
        return traj
    
    def get_traj_from_plan(self, 
        plan, 
        params=None, 
        from_state=None, 
        render=False
    ):
        coas = {role: coa for role, coa in plan.items()}

        agent_dict = dict()
        for role, model_class in plan.model_classes.items():
            agent_dict[role] = model_class(role, **plan.model_params[role])

        traj = self.simulate(
            coas,
            parameters=params,
            agents=agent_dict,
            state=from_state,
            render=render
            )

        return traj

    def trajectory_telemetry(
        self,
        traj, 
        components=None, 
        labels=None, 
        players=None, 
        ao="observations",
        plan=None,
        env=None,
        cmap=None
    ):

        if players is None:
            players = list(traj.step_returns[0][0].keys())

        if cmap is None:
            default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = lambda i: default_cycle[i%len(default_cycle)]

        traj_len = len(traj)

        rewards = Telemetry(
            name="Reward", 
            title="Reward", 
            xscale = np.array(range(traj_len)),
            xlabel = "Time Step",
            ylabel = "Reward",
        )

        rewards.set_data(np.zeros([len(players),traj_len]))

        for i, p in enumerate(players):
            rewards.data_labels[i] = p
            rewards.colors[i] = cmap(i)

            for j, t in enumerate(traj.step_returns[1:]): # No intial reward
                rewards.data[i, j] = t[1][p]

        return [rewards]

    # Note: This should be able to handle plotting 
    # without having to have the origional env spun up. 
    def plot_trajectory_component(
            self,
            traj, 
            components=[0], 
            labels=None, 
            players=None, 
            ao="observations",
            plan=None,
            env=None,
            render_mode="rgb_array" # Union["rgb_array", "matplotlib", "plotly"]
        ):

        if not players:
            players = list(traj.step_returns[0][0].keys())

        if not type(players) is list:
            players = [players] 

        create_labels = False
        if not labels:
            labels = dict()
            create_labels = True

        if ao == "observations":
            data = {}
            all_components = False
            if not components:
                all_components = True

            for p in players:
                action_list = []
                for t in traj.step_returns:
                    t = t[0]
                    if all_components:
                        components = list(range(len(np.array(t[p]).reshape(-1))))

                    action_list.append(t[p][components])

                data[p] = np.array(action_list)
                if create_labels:
                    labels[p] = [f"{p}: {c}" for c in components]

        if ao == "actions":
            data = {}
            all_components = False
            if not components:
                all_components = True

            for p in players:
                action_list = []
                for t in traj.actions[1:]:
                    if all_components:
                        components = list(range(len(np.array(t[p]).reshape(-1))))

                    action_list.append(t[p][components])

                data[p] = np.array(action_list)
                if create_labels:
                    labels[p] = [f"{p}: {c}" for c in components]


        if ao == "rewards":
            components = [0]
            data = traj.step_returns
            xs = np.zeros([len(players), len(components), len(data)-1])

            for j, t in enumerate(data):
                if j > 0: # The initial state has no reward
                    for i, p in enumerate(players):
                        xs[i, :, j] = t[1][p]

        n_axes = sum([s.shape[1] for s in data.values()])
        # Setup Figures
        w = n_axes // 2
        w_m = w + n_axes % 2

        f = plt.gcf()
        axes = []
        for i in range(w):
            for j in range(2):
                axes.append(plt.subplot2grid((w_m, 4), (i, 2 * j), 1, 2))

        if n_axes % 2:
            axes.append(plt.subplot2grid((w + 1, 4), (w, 1), 1, 2))


        # Get Data

        idx = 0
        for p in players:
            for j in range(data[p].shape[1]):
                axes[idx].plot(data[p][:,j])
                axes[idx].title.set_text(labels[p][j])
                idx += 1

        # l5 = axes[0].legend(
        #     bbox_to_anchor=(0.5, -0.05), loc="lower center", bbox_transform=f.transFigure
        # )

        plt.subplots_adjust(left=0.1, right=0.9, hspace=0.3, wspace=0.5)
        
        fig = plt.gcf()
        if render_mode == "matplotlib":
            return fig

        canvas = FigureCanvas(fig)
        canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        if render_mode == "rgb_array":
            return data
        
        if render_mode == "plotly":
            import plotly.express as px 
            fig = px.imshow(data)
            return fig





#################################################################
## Example Usage
#################################################################

if __name__ == "__main__":
    from examples.MAInspection.env import MultInspect

    ## Environemntal Parameters
    env_params = {"_OBS_REWARD":.01, "num_deputes": 3}

    # Actor Parameters
    COACH_params = {
        "Agents": {
            "player_0": {
                "class_name":"DefaultActor"
                # params: {"parameter": "value"}    # If the agent has setup parameters
                },
            "player_1": {"class_name":"RandomActor"},
            "player_2": {"class_name":"DefaultActor"},
        }
    }

    # Create actors from parameters
    agent_dict = dict()
    if "Agents" in COACH_params.keys():
        for role, agent in COACH_params["Agents"].items():
            agent_class = COACHEnvironment.agent_selection.Agents[agent["class_name"]]
            agent_dict[role] = agent_class(role, **agent.get("params", dict()))

    # Example communication scheudle. 
    comms = CommunicationSchedule.repeating(checkin_frequency=10)

    # Wrap parameters in parameter generator
    parameter_generator = SeededParameterGenerator(23423, env_params)

    env = COACHEnvironment(
        env_creator = MultInspect,
        default_parameters = parameter_generator,
        comm_schedule = comms,
        fill_random = True,
        agents = agent_dict
    )

    for agt in env.agents.values():
        print(agt.interface)
    input()

    for i in range(10):
        print(env.step())
        for role, actions in env.action_spaces.items():
            print(f"{role}\t {actions}")
        input("Press Enter For Next Observation")