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

"""Multi-Inspection Environment"""

import argparse  # for type hinting
import copy
import logging
from collections.abc import Callable  # for type hinting
from typing import Any, Optional, TypeVar, Union, cast, NewType

import gymnasium as gym
import imageio
import matplotlib  # type: ignore[import]
import matplotlib.lines as mlines  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np

# from pettingzoo.utils.env import ParallelEnv
from matplotlib.backends.backend_agg import (
    FigureCanvasAgg as FigureCanvas,  # type: ignore[import]
)
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # type: ignore[import]
from numpy.random import PCG64DXSM, Generator
from scipy.spatial.transform import Rotation  # type: ignore[import]
from scipy.stats import multivariate_normal, ortho_group  # type: ignore[import]

import sys

from numpy.typing import NDArray
FloatArray = NDArray[np.float_]
RenderFrame = NDArray[np.uint8]
Role = NewType("Role", str)

from astropy import units as u
from poliastro.maneuver import Maneuver
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit

matplotlib.use("agg")
logger = logging.getLogger(__name__)

from pettingzoo import ParallelEnv

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
    return MultInspect

def unit(v):
    if len(v.shape) == 1:
        v = v.reshape(1,-1)
    return v/np.linalg.norm(v,axis=1)[:,None]

def proj(v, onto):
    return onto * np.dot(v, onto) / np.dot(onto, onto)

def frame(r, v):
    r = unit(r)
    v = unit(v)
    n = np.cross(r,v, axis=1)

    return np.stack([r,v,n],axis=2)

def parallel_env(**kwargs):
    return MultInspect(**kwargs)

class MultInspect(ParallelEnv):
    """
    We assuime six axis movement:
        action space: [X-Thrust, Y-Thrust, Z-Thrust,
                       X-Clockwise Torque, Y-Clockwise Torque, Z-Clockwise Torque]
    
    Some notes on linear algebra conventions and frames:

    All frames have columns as vectors in the background coordinate system R, so
    for any frame M, M * v maps v to R. To traslate back, a vector in w in R is mapped to 
    M^T * w.

    A good rule of thumb here is to understand every matrix as having units
    xf, where x is standard coords and f is frame coords. Since frames are orthonormal
    M^-1 = M^T. If we have a vcetor in frame 1 coords and we want it in frame 2
    coords, we can just pass through the standard coords: M_1: v_f1 -> v_x, 
    M_2^T: v_x -> v_f2, so M_2^T M_1 * v_f1 = v_f2 is a appropreate transform

    A source for physics: http://control.asu.edu/Classes/MMAE441/Aircraft/441Lecture9.pdf

    We're implementing a Petting Zoo interface, details can be found here:
    https://pettingzoo.farama.org/api/parallel/#parallelenv


    Attributes
    -------------

    agents: list[AgentID]
        A list of the names of all current agents, typically integers. May changed as environment progresses
    num_agents: int
        The length of the agents list.
    possible_agents: list[AgentID]
        A list of all possible_agents the environment could generate. Equivalent to the list of agents in the observation and action spaces. This cannot be changed through play or resetting.
    max_num_agents: int
        The length of the possible_agents list.
    observation_spaces: dict[AgentID, gym.spaces.Space]
        A dict of the observation spaces of every agent, keyed by name. This cannot be changed through play or resetting.
    action_spaces: dict[AgentID, gym.spaces.Space]
        A dict of the action spaces of every agent, keyed by name. This cannot be changed through play or resetting.


    Methods
    -------------

    step(actions: dict[str, ActionType]) → tuple[dict[str, ObsType], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]
        Receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary and info dictionary, where each dictionary is keyed by the agent.
    reset(seed: int | None = None, options: dict | None = None) → dict[str, ObsType]
        Resets the environment.
    seed(seed=None)
        Reseeds the environment (making it deterministic).
    render() → None | np.ndarray | str | list
        Displays a rendered frame from the environment, if supported.
    close()
        Closes the rendering window.
    state() → ndarray
        Returns the state.
    observation_space(agent: str) → Space
        Returns the observation space for given agent.
    action_space(agent: str) → Space
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50, "name": "MAInspect"}


    def __init__(self, render_mode="rgb_array", **kwargs) -> None:
        """ """
        self.augment(params = kwargs)
        self.render_mode = render_mode

    def parallel_env(self, **kwargs):
        return self

    def _setup_gym_env(self, env_params: dict) -> None:
        ## Setup parameters. All of this may be overwritten by the
        ## params dictionary if the appropreate key exists.
        ## The list below provides the default parameters

        self.verbose: bool = False
        self.six_axis: bool = True
        self.num_deputies: int = 3
        self.max_episode_steps: int = 500
        self.num_points: int = 20

        self.MAXTIME = 1 << u.h
        self.TIME_PER_STEP = 1 << u.min

        # Compatability
        # This disable some agents terminating before others. 
        self._SB_SAFTY_MODE = True

        # Game Mode:
        self._TRAIN_WAYPOINTER = True
        self._WAYPOINT_ARRIVAL_PROX = 10 << u.m
        self._WAYPOINT_ARRIVE_REWARD = 1

        # Initial Conditions
        self.INIT_ALT = 700 << u.km
        self.INIT_ALT_OFFSET = 50 << u.m
        self.RAAN = 0 << u.deg ## Angle Around the Circular Orbit
        self.ARGLAT = 0 << u.deg ## Latitude updown
        self.INC = 0 << u.deg ## Direction of orbit
        self._STARTING_DISTANCE_FROM_CHIEF = 150 << u.m

        self.offset_angle = (1/10)*(self._STARTING_DISTANCE_FROM_CHIEF / self.INIT_ALT).decompose().value << u.rad

        # Action Frame
        self.ACTION_FRAME = "Hills"         # Options should be Hills and Orientation
        self.OBSERVATION_FRAME = "Hills"    # Options should be Hills and Orientation
        self._OU_DIS = u.m 
        self._OU_TIM = u.s 
        self._OU_VEL = self._OU_DIS/self._OU_TIM

        self._CHIEF_PERIMETER: float = 50 << u.m
        self._DEPUTY_PERIMETER: float = 150  << u.m
        self._DEPUTY_MASS: float = 1 << u.kg
        self._DEPUTY_RADIUS: float = 1 << u.m
        self._DEPUTY_THRUST_COEEF: float = 0.01

        self._SIM_OUTER_PARAMETER: float = 200 << u.m #
        self._MAX_OUTER_PERIMETER: float = 300 << u.m # If you leave here, you truncate
        self._STARTING_VEL_NORM: float = 0 << u.m/u.s # 0.001

        self._TIME_PER_STEP: float = 90 << u.s
        self._DELTA_V_PER_THRUST: float = 0.1 << u.m/u.s
        self._DELTA_T: float = 90 << u.s
        self._TAU_per_THRUST: float = 0.1 
        self._USE_ANGULAR_MOMENTUM = False

        self._OBS_REWARD: float = 0.02  # 20
        self._REWARD_FOR_SEEING_ALL_POINTS: float = 1  # 200
        self._CRASH_REWARD: float = 0
        self._REWARD_PER_STEP: float = 0.0   # -.1  # Reward per tick
        self._REWARD_FOR_LEAVING_PARAMETER: float = 0
        self._REWARD_FOR_LEAVING_OUTER_PERIMETER: float = 0
        
        self._SOLID_CHIEF: bool = True
        self._MAX_BURN: float = 10

        self._REW_COV_MTRX = 300**2
        self._PROX_RWD_SCALE = 100000

        # Visualization
        self._VIS_NUM_CONE_LINES = 8
        self._VIS_CONE_LEN = 40 << u.m

        # Vision Cone:
        self._MIN_VISION: float = 0 << u.m
        self._MAX_VISION: float = np.inf << u.m
        self._VISION_ARC: float = np.pi / 8 << u.rad

        # Apply parameters from input
        self._apply_parameters(params=env_params)
        if hasattr(env_params,"args"):
            self.master_seed: int = env_params.args.master_seed
        else:
            self.master_seed: int = 487924
        self.seed(seed=self.master_seed)


        if self._TRAIN_WAYPOINTER:
            self.num_points = self.num_deputies

        # Check parameters from input
        assert self._MIN_VISION >= 0, "_MIN_VISION must be >= 0"
        assert self._MAX_VISION > self._MIN_VISION, "_MAX_VISION must be > _MIN_VISION"
        assert (self._VISION_ARC >= 0) and (
            self._VISION_ARC <= np.pi << u.rad
        ), "_VISION_ARC must be [0, pi]"

        ## Setup actual Gym Env based on the above
        self.agents: list[Role] = [Role(f"player_{i}") for i in range(self.num_deputies)]

        # Agent Velocity, Other Dep. Rel Position and Vel, Point Rel Position
        self._VISION: dict[Role, list[float]] = {
            role: [self._VISION_ARC, self._MIN_VISION, self._MAX_VISION]
            for role in self.agents
        }

        in_size: int = (
            6 + 6 * (self.num_deputies - 1) + 3 * self.num_points + self.num_points
        )
        
        if self._TRAIN_WAYPOINTER:
            raw_low_values: list[float] = [
                -self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                -self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                -2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                -2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3),  # Single Waypoint,
            ]

            raw_high_values: list[float] = [
                self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3),  # Single Waypoint,
            ]
        else:
            raw_low_values: list[float] = [
                -self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                -self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                -2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                -2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_points)),  # Point Pos,
                np.zeros(self.num_points),
            ]

            raw_high_values: list[float] = [
                self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_points)),  # Point Pos,
                np.ones(self.num_points),
            ]

        low = np.concatenate(raw_low_values, dtype=np.float32)
        high = np.concatenate(raw_high_values, dtype=np.float32)

        obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if self.six_axis:
            low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi], dtype=np.float32)
            act_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)
        else:
            # Thrust, rotate Left/Right, rotate UP/Down
            act_space = gym.spaces.Box(
                low=-np.ones(3, dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                dtype=np.float32,
            )

        self.possible_agents = self.agents

        self.observation_spaces = {role: obs_space for role in self.possible_agents}
        self.action_spaces = {role: act_space for role in self.possible_agents}

        self.ori = np.zeros([self.num_deputies, 3, 3])       # Depute Orientation relative to absolute earth frame
        self.rot = np.zeros([self.num_deputies, 3])          # Depute Angular Momentum

        self.pos = np.zeros([self.num_deputies, 3])          # Depute Position Holder
        self.vel = np.zeros([self.num_deputies, 3])          # Depute Velocity Holder

        self.chief_frame = np.zeros([3, 3])                 # Current frame for the cheif relative to absolute earth frame      
        self.frames = np.zeros([self.num_deputies, 3, 3])    # Current frame for the deputies relative to absolute earth frame
        self.pts = np.zeros([self.num_deputies, 3])          # Chief Inspection Points
        self.nor = np.zeros([self.num_deputies, 3])          # Chief Inspection Normals

        self.sim_steps = 0

        self.unobserved_points = set(range(self.num_points))

        self.render_path: Optional[str] = None
        self.frame_store: list[RenderFrame] = []
        self.cum_rewards = np.zeros(self.num_deputies)

    def _apply_parameters(self, params: dict) -> None:
        for k, v in params.items():
            self.__dict__[k] = copy.copy(v)

    def augment(self, params: dict) -> None:
        self._setup_gym_env(env_params=params)

    def seed(self, seed: int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed))

    def observation_space(self, agent: Role) -> gym.spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: Role) -> gym.spaces.Box:
        return self.action_spaces[agent]

    ## Reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[Any, Any]] = None,
        render_path: Optional[str] = None,
    ) -> tuple[dict[Role, Any], dict[Role, dict[str, Any]]]:
        """ """
        self.sim_steps = 0

        # Reset Seed
        local_seed = seed if (seed is not None) else self.master_seed
        self.seed(seed=local_seed)
        local_og = ortho_group(dim=3, seed=self.np_random)

        # Create Orbits for centers of mass
        self.orb = dict()
        self.orb_hist = []

        # Construct Chief 
        self.orb[f"chief"] = Orbit.circular(Earth, alt=self.INIT_ALT)
        if self._TRAIN_WAYPOINTER:
            self.pts = self._random_sphere(self.num_points, radius=self._DEPUTY_PERIMETER)
        else:
            self.pts = self._random_sphere(self.num_points, radius=self._CHIEF_PERIMETER)

        self.chief_angular_momentum = None
        # self.chief_angular_momentum = Rotation.from_euler("xyz", [1,0,0]).as_matrix()

        # Construct Deputies
        self._active = np.array([role in self.agents for role in self.possible_agents])

        # Deputy Orbits
        for i in range(self.num_deputies):
            self.orb[f"player_{i}"] = Orbit.circular(Earth, alt=self.INIT_ALT + i*self.INIT_ALT_OFFSET, raan = self.offset_angle)

        self.ori = np.stack([local_og.rvs() for i in range(self.num_deputies)])
        self.rot = self._random_sphere(self.num_deputies, radius=self._CHIEF_PERIMETER)

        # Make Initial Observations

        # Compute rewards
        if self._TRAIN_WAYPOINTER:
            obs = self._make_waypoint_observations()
        else:
            obs = self._make_observations()

        self.unobserved_points = set(range(self.num_points))
        self.truncated = {Role(f"player_{i}"): False for i in range(self.num_deputies)}
        
        # Initial relative positions and velocities
        self.pos = np.stack([self.orb[role].r-self.orb["chief"].r for role in self.possible_agents])
        self.vel = np.stack([self.orb[role].v-self.orb["chief"].v for role in self.possible_agents])

        # Setup Reward Structure for waypoints
        self.pt_prox = []
        for i in range(self.num_points):
            self.pt_prox.append(multivariate_normal(mean=self.pts[i], cov=self._REW_COV_MTRX))

        self.cum_rewards = np.zeros(self.num_deputies)

        # Setup Longterm Rendering
        self.render_path = render_path
        if render_path:
            frame = self.render()
            assert frame is not None
            self.frame_store = [frame]

        self.render_data = dict()

        return obs, {agt: dict() for agt in self.possible_agents}

    def step(
        self, action: dict[str, FloatArray]
    ) -> tuple[
        dict[Role, Any],
        dict[Role, float],  # reward
        dict[Role, bool],  # terminated
        dict[Role, bool],  # truncated
        dict[Role, dict[str, Any]],  # info
    ]:
        """ """
        # truncated = self.truncated
        self.terminated_this_step = False

        # Propagate Motion In Time
        self._propagate_objects(action)

        self.pos = np.stack([self.orb[role].r-self.orb["chief"].r for role in self.possible_agents])
        self.vel = np.stack([self.orb[role].v-self.orb["chief"].v for role in self.possible_agents])

        # Compute Which Points Seen
        just_seen = self._detect_points()

        # Compute rewards
        if self._TRAIN_WAYPOINTER:
            self._compute_waypoint_reward(just_seen)
            obs = self._make_waypoint_observations()
        else:
            self._compute_reward(just_seen)
            obs = self._make_observations()

        # Check if all of the deputes have truncated.
        if all(self.truncated.values()):
            terminated = True

        if self.render_path:
            frame = self.render()
            assert frame is not None
            self.frame_store.append(frame)

            if terminated:
                imageio.mimwrite(
                    uri=self.render_path,
                    ims=self.frame_store,
                    fps=int(60 / 10),  # type: ignore[arg-type]
                    loop=0,
                )

        self.cum_rewards += self.sra

        self.obs = obs
        self.reward = {role: rwd[0] for role, rwd in self.step_rewards.items()}

        self.info: dict[Role, dict[str, Any]] = {
            agt: dict() for agt in self.possible_agents
        }

        self.terminated = {agt: self.terminated_this_step for agt in self.possible_agents}

        # Remove Terminated and Truncated agents from avaiable agent lists
        if self._SB_SAFTY_MODE:
            self.agents = self.possible_agents
        else:
            self.agents = []
            for role in self.possible_agents:
                if not (self.terminated[role] or self.truncated[role]):
                    self.agents.append(role)
        

        # Set the internal active players array
        self._active = np.array([role in self.agents for role in self.possible_agents])

        return (self.obs, self.reward, self.terminated, self.truncated, self.info)


    def _compute_reward(self, just_seen: list[list[int]]) -> float:
        self.sra = np.zeros(self.num_deputies) # Step Reward Array
        self.step_rewards = {role: self.sra[i:i+1] for i, role in enumerate(self.agents)} # Slices are pointers to the reward array

        self._insepct_reward(just_seen)
        self._chief_prox_reward()
        self._game_length_exceed()
        self._seeing_all_points()
        self._leaving_outer_perimeter()


    def _compute_waypoint_reward(self, just_seen: list[list[int]]) -> float:
        self.sra = np.zeros(self.num_deputies) # Step Reward Array
        self.step_rewards = {role: self.sra[i:i+1] for i, role in enumerate(self.agents)} # Slices are pointers to the reward array
        
        self._go_towards_waypoint()
        self._game_length_exceed()
        self._leaving_outer_perimeter()
        self._chief_prox_reward()


    def _go_towards_waypoint(self):
        # We're going to change the reward structure to include waypointing
        for i, role in enumerate(self.agents):
            self.step_rewards[role] += self.pt_prox[i].pdf(self.pos[i]) * self._PROX_RWD_SCALE
            if np.linalg.norm(self.pos[i] - self.pts[i]) < self._WAYPOINT_ARRIVAL_PROX:
                self.step_rewards[role] += self._WAYPOINT_ARRIVE_REWARD
                self.terminated_this_step = True
            


    def _seeing_all_points(self):
        if len(self.unobserved_points) == 0:
            R = (self.max_episode_steps - self.sim_steps) / self.max_episode_steps
            self.sra += R * self._REWARD_FOR_SEEING_ALL_POINTS
            self.terminated_this_step = True



    def _leaving_outer_perimeter(self):
        I = np.where(np.linalg.norm(self.pos, axis=1) > self._MAX_OUTER_PERIMETER)[0]

        # print(np.linalg.norm(self.pos, axis=1) > self._MAX_OUTER_PERIMETER)
        # print(I, self.possible_agents, self.pos)
        for i in I:
            role = Role(f"player_{i}")
            if not self.truncated[role]:
                self.truncated[role] = True
                self.step_rewards[role] += self._REWARD_FOR_LEAVING_OUTER_PERIMETER



    def _game_length_exceed(self):
        self.sim_steps += 1
        if self.sim_steps > self.max_episode_steps:
            ## Technically we were truncated
            self.truncated = {
                Role(f"player_{i}"): True for i in range(self.num_deputies)
            }
            self.terminated_this_step = True



    def _insepct_reward(self, just_seen):
        """ """
        for i, role in enumerate(self.possible_agents):
            seen_points = just_seen[i]
            # Initialize reward
            #  Start with the survival reward
            reward: float = self._REWARD_PER_STEP

            # Check if seen points are new
            new_pts: set = self.unobserved_points.intersection(seen_points)

            # Check if new points were seen
            if new_pts:
                # Reward for seeing points for the first time
                reward += len(new_pts) * self._OBS_REWARD
                # Update unseen points
                self.unobserved_points = self.unobserved_points.difference(new_pts)

            self.step_rewards[role] += reward

        

    def _chief_prox_reward(self):
        # Are we active and outside the safe zone?
        outside_safe_zone = self._active & (np.linalg.norm(self.pos, axis=1) > self._SIM_OUTER_PARAMETER)
        # Are we active and inside the chief?
        inside_chief = self._active & (np.linalg.norm(self.pos, axis=1) < self._CHIEF_PERIMETER)

        # Adjust reward
        self.sra[outside_safe_zone] += self._REWARD_FOR_LEAVING_PARAMETER
        self.sra[inside_chief] += self._CRASH_REWARD

        # Kill agents inside chief 
        self._active[inside_chief] = False

        for i in np.where(inside_chief)[0]:
            self.terminated_this_step = True


    def _random_sphere(self, n_points: int, radius: float) -> FloatArray:
        """ """
        ## Generate random points on a sphere
        pts: FloatArray = -1 + 2 * self.np_random.normal(size=[n_points, 3])
        pts = pts / np.linalg.norm(pts, axis=1).reshape(-1, 1)
        pts = pts * radius

        return pts

    # Physics goes here
    def _propagate_objects(self, actions):
        # Poliastro assumes earth centered coordinates. It fixes an x,y, and z. So  
        # vectors in poliastro objects are in terms of of an absolute coordinate system.
        
        # Translate actions into action frame and apply impulses
        
        for player, (role, act) in enumerate(actions.items()):
            # Compute Velocity Change
            dv = act[:3].reshape(-1,1) * self._DELTA_V_PER_THRUST
            if self.ACTION_FRAME == "Hills":
                frame = self.hills_frame(self.orb[role])
            else:
                frame = self.ori[role]

            imp = Maneuver.impulse((frame @ dv).reshape(-1))
            self.orb[role] = self.orb[role].apply_maneuver(imp)

            # Apply rotations
            # Sanity Check: if frame is ori, than an action of [1,0,0]
            # fixes ori[:,0] and rotates ori[:,1] and ori[:,2]
            # Here, R has units ff, mapping from frame coords to frame coords
            # with left multiplaction assumed for standard clockwise rotation

            tau = self._TAU_per_THRUST * act[-3:]


            if self._USE_ANGULAR_MOMENTUM:
                self.rot[player] += tau
                tau = self.rot

            R = Rotation.from_euler("xyz", tau)
            T = frame @ R.as_matrix() @ frame.T # x columns to frame, rotate, then back to x

            self.ori[player, :, :] = T @ self.ori[player, :, :]
        
        # Rotate Chief Points:
        if self.chief_angular_momentum is not None:
            R = self.chief_angular_momentum
            self.pts = R @ self.pts

        # Propagate Orbits
        for role, orb in self.orb.items():
            self.orb[role] = orb.propagate(self._TIME_PER_STEP)

        # for role in self.possible_agents:
        #     print(role, self.orb[role].r, np.linalg.norm(self.orb[role].r - self.orb["chief"].r))

    def _detect_points(self):
        just_seen = []
        for player, role in enumerate(self.possible_agents):
            # Position relvative to chief
            pos = self.orb[role].r - self.orb["chief"].r

            # Detect Points:
            #  This handles occlusion by the chief
            #  Basically, what points are on the same side of the chief as you
            seeable_points = np.where(
                (self.pts * (pos - self.pts)).sum(axis=1) > 0
            )[0]

            ##########
            # Spherical Vision
            #########
            # point holder
            seen_points: list[int] = []

            # loop over possible points
            for point in seeable_points:
                # Cone References
                # cone: https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space
                #  We're going to not use the cone, and switch to using a shell
                #  The initial distance calc though is fine, and easy
                #
                # Shell References
                #  This is a pain in the a$$ - I hate math vs physics
                #  Used all of these references to generate a consensus algorithm
                #  https://en.wikipedia.org/wiki/Spherical_coordinate_system
                #   This one doesn't fully work
                #  https://en.wikipedia.org/wiki/Atan2
                #   This explains why atan2 vs atan
                #  https://mathworld.wolfram.com/SphericalCoordinates.html
                #  https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
                #   THIS IS MY FAVORITE ONE - uses physics notation
                #  https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.07%3A_Cylindrical_and_Spherical_Coordinates
                #   This supports the stackoverflow, just in math notation

                # Position relative to agent
                v = self.pts[point, :] - pos

                # Distance relative to agent
                #  this is the same as the distance in the deputy frame, but
                #  doesn't require the transformation
                p_dist = np.linalg.norm(v)

                # Check if distance is within shell
                #  Ignoring angles at the moment
                #  Less than max, more than min
                #   I assume we're usually too far away
                if (p_dist < self._VISION[role][2]) and (
                    p_dist >= self._VISION[role][1]
                ):
                    # Orient point in deputy frame
                    pdf = self.ori[player].T @ v

                    # theta and phi
                    #  This is physics notation
                    #  theta = polar/zenith angle, [0, pi], z-axis is 0
                    #  phi = azimuth angle, (-pi, pi], x-axis is 0
                    theta = np.arccos(pdf[2] / p_dist)
                    phi = np.arctan2(pdf[1], pdf[0])

                    # theta needs to measure from x-axis, not z-axis
                    #  calculate the complementary angle
                    #  theta = [pi/2, -pi/2], x-axis is 0
                    theta = (np.pi / 2 << u.rad) - theta

                    # Check if point is within cone
                    #  Assume circular directions, so abs() it
                    # NOTE: We could implement different ranges for theta and phi
                    #       This would give a non-circular viewing angle
                    if (np.abs(phi) < self._VISION[role][0]) and (
                        np.abs(theta) < self._VISION[role][0]
                    ):
                        # you can see it!
                        seen_points.append(point)
            
            just_seen.append(seen_points)
            if self.verbose:
                print(f"{player} seeable: {seeable_points}, seen: {seen_points}")
        
        return just_seen
    
    def accel(self, t0, state, k, rate=1e-5):
        """Constant acceleration aligned with the velocity. """
        v_vec = state[3:]
        #v_vec = state[:3]
        norm_v = (v_vec * v_vec).sum() ** 0.5
        return -rate * v_vec / norm_v

    def f(self, t0, u_, k):
        # t_0: time to evaluate at
        # u_ = [x,y,z,vx,vy,vz] in earth coords, rather annoyingly unitless. 
        # Assumed units (I've tested this) are km and km/s

        # U.append(u_)
        du_kep = func_twobody(t0, u_, k)
        #ax, ay, az = self.accel(t0, u_, k, rate=1e-5)
        ax, ay, az = self.acc
        du_ad = np.array([0, 0, 0, ax, ay, az])
        return du_kep + du_ad

    def hills_frame(self, orb):
        r = orb.r
        v = orb.v
        v_p = v - proj(v, onto=r)
        return frame(r,v_p)[0].decompose().value

    def _make_observations(self, obs_frame=None) -> dict[Role, FloatArray]:
        """ """
        # Return dictionary
        # Play role will be keys, observations will be values
        obs = dict()
        
        if obs_frame is None:
            obs_frame = self.OBSERVATION_FRAME

        # Setup point mask
        #  hide points that have been observed
        observedPoints = list(set(range(self.num_points)) - self.unobserved_points)
        pt_mask = np.ones(self.num_points)
        pt_mask[observedPoints] = 0

        poss = np.stack([self.orb[role].r for role in self.possible_agents]).T.to(self._OU_DIS).value
        vels = np.stack([self.orb[role].v for role in self.possible_agents]).T.to(self._OU_VEL).value
        pos_c = self.orb["chief"].r.reshape(-1,1).to(self._OU_DIS).value
        vel_c = self.orb["chief"].v.reshape(-1,1).to(self._OU_VEL).value
        pos_p = self.pts.T.to(self._OU_DIS).value

        # loop over all deputies
        for i, role in enumerate(self.possible_agents):
            # Recall: Moving from absolute coords into a frame is left multiplacation
            # by the transpose

            # Velocities
            # Velocities of all deputies in my frame
            if obs_frame == "Hills":
                frame = self.hills_frame(self.orb[role])
            else:
                frame = self.ori[i] 

            vel_in_frame = frame.T @ (vels - vels[:,[i]])
            chief_vel_in_frame = frame.T @ (vel_c - vels[:,[i]])

            #  Relative velocities of all other deputies in my frame
            rel_vel = np.delete(arr=vel_in_frame, obj=i, axis=1)

            # Relative positions of all other deputies in my frame
            rel_pos = (
                frame.T @ np.delete(arr=poss - poss[:,[i]], obj=i, axis=1)
            )

            dchief = pos_c - poss[:,[i]]

            # Relative positions of all points in my frame
            rel_pts = frame.T @ (pos_p + dchief)

            # Relative position of chief in my
            rel_chief = frame.T @ dchief

            # Put the observation space together
            tmp = np.concatenate(
                [   
                    rel_chief, 
                    chief_vel_in_frame, 
                    rel_vel,
                    rel_pos, 
                    rel_pts,
                ], axis=1
            )

            # Store observation space under appropriate role
            obs[Role(f"player_{i}")] = np.concatenate([tmp.reshape(-1), pt_mask])

        return obs

    def _make_waypoint_observations(self, obs_frame=None) -> dict[Role, FloatArray]:
        """ """
        # Return dictionary
        # Play role will be keys, observations will be values
        obs = dict()
        
        if obs_frame is None:
            obs_frame = self.OBSERVATION_FRAME

        # Setup point mask
        #  hide points that have been observed
        observedPoints = list(set(range(self.num_points)) - self.unobserved_points)
        pt_mask = np.ones(self.num_points)
        pt_mask[observedPoints] = 0

        poss = np.stack([self.orb[role].r for role in self.possible_agents]).T.to(self._OU_DIS).value
        vels = np.stack([self.orb[role].v for role in self.possible_agents]).T.to(self._OU_VEL).value
        pos_c = self.orb["chief"].r.reshape(-1,1).to(self._OU_DIS).value
        vel_c = self.orb["chief"].v.reshape(-1,1).to(self._OU_VEL).value
        pos_p = self.pts.T.to(self._OU_DIS).value
        

        # loop over all deputies
        for i, role in enumerate(self.possible_agents):
            # Recall: Moving from absolute coords into a frame is left multiplacation
            # by the transpose

            # Velocities
            # Velocities of all deputies in my frame
            if obs_frame == "Hills":
                frame = self.hills_frame(self.orb[role])
            else:
                frame = self.ori[i] 

            vel_in_frame = frame.T @ (vels - vels[:,[i]])
            chief_vel_in_frame = frame.T @ (vel_c - vels[:,[i]])

            #  Relative velocities of all other deputies in my frame
            rel_vel = np.delete(arr=vel_in_frame, obj=i, axis=1)

            # Relative positions of all other deputies in my frame
            rel_pos = (
                frame.T @ np.delete(arr=poss - poss[:,[i]], obj=i, axis=1)
            )

            dchief = pos_c - poss[:,[i]]

            # print("dchief", dchief)
            # print(self.pts.T.to(self._OU_DIS).value + dchief)
            # Relative positions of all points in my frame
            rel_pts = frame.T @ (pos_p[:,[i]] + dchief)

            # Relative position of chief in my
            rel_chief = frame.T @ dchief

            # Put the observation space together
            tmp = np.concatenate(
                [   
                    rel_chief, 
                    chief_vel_in_frame, 
                    rel_vel, 
                    rel_pos, 
                    rel_pts,
                ], axis=1
            )

            # Store observation space under appropriate role
            obs[Role(f"player_{i}")] = tmp.reshape(-1)

        return obs



    ## Visualizations
    def render(
        self,
        render_mode: str = "rgb_array",
        vision: Optional[int] = 0,  # player number, but not their role, just the number
        rotate: bool = False,
        elev: float = 45,  # I think it's degrees?
        close: bool = False,
        **kwargs: Any,
    ) -> Union[RenderFrame, None]:
        """ """

        if close:
            self.close()
            return None

        # init figure and canvas
        fig = plt.figure(figsize=[20,5])
        canvas = FigureCanvas(fig)

        poss = np.stack([self.orb[role].r-self.orb["chief"].r for role in self.possible_agents]).T.to(self._OU_DIS).value
        vels = np.stack([self.orb[role].v-self.orb["chief"].v for role in self.possible_agents]).T.to(self._OU_VEL).value

        ##########
        # Title
        ##########
        # plot scores as title
        score_title = "    ".join(
            [f"Player {i}: {self.cum_rewards[i]:.3f}" for i in range(self.num_deputies)]
        )
        fig.suptitle(
            t=score_title, horizontalalignment="center", verticalalignment="center"
        )

        ##########
        # Setup Plot Area
        ##########
        # Plot just the chief, or the chief and one agent-centric view?
        if vision is not None:
            ax = fig.add_subplot(141, projection="3d")
        else:
            ax = plt.axes(projection="3d")

        ##########
        # First Sub Area
        ##########
        # Orientation
        deg = 45
        if (vision is not None) and rotate:
            x, y = poss[vision, 0:2] / np.linalg.norm(poss[vision, 0:2])
            deg = np.degrees(np.arctan2(y, x))

        ax.view_init(elev=elev, azim=deg)

        # limits
        # ax.set_xlim([-300, 300])
        # ax.set_ylim([-300, 300])
        # ax.set_zlim([-300, 300])

        ##########
        # Deputies
        ##########
        # I think all of this just plots deputies - JB

        # Deputy body
        for i in range(self.num_deputies):
            x, y, z = poss[:,[i]]
            ax.plot3D(x, y, z, marker="^", linewidth=0, label=f"player_{i}")

        # Velocity
        # Plot line collection: LC = np.array([N,S,D]), num lines, num of points
        # per line, dim of space, so LC[2,0,:] is point 0 on line 2


        TPS = (self.TIME_PER_STEP * self._OU_VEL/self._OU_DIS).decompose().value

        segs = np.stack([poss.T, poss.T + TPS * vels.T], axis=1)
        line_segments = Line3DCollection(
            segs, linestyle="solid", label="Velocity", color="k"
        )
        ax.add_collection(line_segments)

        # Orientation
        colors = ["r", "g", "b"]
        labels = ["Roll/Heading", "Yaw", "Pitch"]
        for axis in range(len(labels)):
            segs = np.stack([poss.T, poss.T + 30 * self.ori[:, axis, :]], axis=1)
            line_segments = Line3DCollection(
                segs, linestyle="solid", color=colors[axis], label=labels[axis]
            )
            ax.add_collection(line_segments)

        ##########
        # Points on Chief
        ##########
        # Shift for plotting
        view_vec = 100 * np.ones(3) / np.sqrt(3)

        # Unobserved points
        if len(self.unobserved_points) > 0:
            # get positions of points
            unobserved_pos = self.pts[list(self.unobserved_points), :]
            # calc vector for point size
            s = np.linalg.norm(unobserved_pos.to(self._OU_DIS).value + view_vec, axis=1) - 40
            # get in euclidean space
            x, y, z = unobserved_pos.T
            # plot
            ax.scatter(x, y, z, label="Unseen", s=s, color="m")
        else:
            # plot for legend
            #  60: comes from s, when unobserved_points is the empty set
            ax.scatter([], [], [], label="Unseen", s=60, color="m")

        # Observed points
        observed_pts = set(range(self.num_points)) - self.unobserved_points
        if len(observed_pts) > 0:
            # get positions of points
            observed_pos = self.pts[list(observed_pts), :]
            # calc vector for point size
            s = np.linalg.norm(observed_pos.to(self._OU_DIS).value + view_vec, axis=1) - 40
            # get in euclidean space
            x, y, z = observed_pos.T
            # plot
            ax.scatter(x, y, z, color="c", s=s, label="Seen")
        else:
            # plot for legend
            #  60: comes from s, when observed_points is the empty set
            ax.scatter([], [], [], color="c", s=60, label="Seen")

        ##########
        # Legend
        ##########
        # get legend handles
        # handles, _ = ax.get_legend_handles_labels()

        # Style
        fig.legend(loc="lower center", ncol=3, fancybox=True)  # , handles=handles

        ##########
        # Chief Sphere
        ##########
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]  # type: ignore[misc]
        x = self._CHIEF_PERIMETER * np.cos(u) * np.sin(v)
        y = self._CHIEF_PERIMETER * np.sin(u) * np.sin(v)
        z = self._CHIEF_PERIMETER * np.cos(v)
        ax.plot_wireframe(x, y, z, color="k", alpha=0.2)

        mins = np.min(poss.T, axis=0) - 10
        maxes = np.max(poss.T, axis=0) + 10

        widths = []
        centers = []
        for i, s in enumerate(['x', 'y', 'z']):
            lim = getattr(ax, f"get_{s}lim")()
            lim = [
                min(max(mins[i], -300), -self._CHIEF_PERIMETER.to(self._OU_DIS).value+10), 
                max(min(maxes[i], 300), self._CHIEF_PERIMETER.to(self._OU_DIS).value+10)
            ]
            widths.append(lim[1]-lim[0])
            centers.append((lim[1]+lim[0])/2)
        
        w = max(widths)/2
        for i, s in enumerate(['x', 'y', 'z']):
            getattr(ax, f"set_{s}lim")([centers[i] - w, centers[i] + w])


        ##########
        # Render Agent View
        ##########
        if vision is not None:
            ax2 = fig.add_subplot(142, projection="3d")
            all_obs = self._make_observations(obs_frame="Orientation")
            role = cast(Role, f"player_{vision}")
            obs = all_obs[role]

            mask = obs[-self.num_points :]
            obs = obs[:-self.num_points].reshape(3,-1)
            x = obs[:,-self.num_points:]

            # Orientation
            z = np.zeros(2)
            o = 30 * np.array([0, 1])
            ax2.plot(o, z, z, color="r")
            ax2.plot(z, o, z, color="g")
            ax2.plot(z, z, o, color="b")



            seen = x[:, mask == 0]
            unseen = x[:, mask == 1]
            ax2.scatter(seen[0], seen[1], seen[2], color="c")
            ax2.scatter(unseen[0], unseen[1], unseen[2], color="m")
            ax2.set_xlim([-100, 200])
            ax2.set_ylim([-100, 200])
            ax2.set_zlim([-100, 200])


        # Render Earth Frame

        ax3 = fig.add_subplot(143)
        
        N = len(self.orb.keys())

        poss = np.stack([orb.r for orb in self.orb.values()]).to(self._OU_DIS).value
        
        if "pos_hist" not in self.render_data.keys():
            self.render_data["pos_hist"] = []

        self.render_data["pos_hist"].append(poss)
        pos_hist = np.array(self.render_data["pos_hist"])

        for i in range(N):
            ax3.plot(pos_hist[:, i, 0], pos_hist[:, i, 1])


        ax3.set_title("Earth Centered Frame")


        # Render Hill Frame

        ax4 = fig.add_subplot(144)

        if "frames" not in self.render_data.keys():
            self.render_data["frames"] = []
    
        if "chief_rel_poss" not in self.render_data.keys():
            self.render_data["chief_rel_poss"] = []

        c_idx = list(self.orb.keys()).index("chief")
        frame = self.hills_frame(self.orb["chief"])

        chief_rel_poss = (frame.T @ poss.T).T
        chief_rel_poss = chief_rel_poss - chief_rel_poss[c_idx,:]

        self.render_data["chief_rel_poss"].append(chief_rel_poss)
        self.render_data["frames"].append(frame)

        hills = np.array(self.render_data["chief_rel_poss"])

        for i in range(N):
            ax4.plot(hills[:, i, 1], hills[:, i, 0])
        
        chief = plt.Circle((0, 0), self._CHIEF_PERIMETER.to(self._OU_DIS).value, color='k', fill=False)
        ax4.add_patch(chief)
        ax4.set_title("Chief Centered Hill Frame")

        # Draw

        if render_mode == "rgb_array":
            canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            return data

        plt.close(fig)
        return None

    def close(self) -> None:
        pass

    def state(self) -> None:
        raise NotImplementedError("State Not Implemented.")


if __name__ == "__main__":
    from tqdm import tqdm
    from PIL import Image

    env = MultInspect(
        num_deputies=4, 
        _SIM_OUTER_PARAMETER=100<<u.m,
        _CHIEF_PERIMETER = 100 << u.m
    )
    env.seed(0)
    env.reset()
    
    frames = []
    # for i in tqdm(range(env.max_episode_steps)):
    for i in tqdm(range(10)):
        act = {role: np.array([0,0,0,0,0,0]) for role in env.possible_agents}
        env.step(act)
        frames.append(env.render())
        
    # Image.fromarray(frames[0])

    imgs = [Image.fromarray(frame) for frame in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    model_dir = ""
    imgs[0].save(model_dir + f"_eval_run_{i}.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)
# %%
