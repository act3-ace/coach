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

# import sys
from collections.abc import Callable  # for type hinting
from typing import Any, NewType, Optional, Union, cast  # TypeVar,

import gymnasium as gym
import imageio

# import matplotlib  # type: ignore[import]
# import matplotlib.lines as mlines  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from astropy import units as u

# from pettingzoo.utils.env import ParallelEnv
from matplotlib.backends.backend_agg import (
    FigureCanvasAgg as FigureCanvas,  # type: ignore[import]
)

# from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # type: ignore[import]
from numpy.random import PCG64DXSM, Generator
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from poliastro.bodies import Earth  # Mars,; Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from scipy.spatial.transform import Rotation  # type: ignore[import]
from scipy.stats import multivariate_normal, ortho_group  # type: ignore[import]

# matplotlib.use("agg")
logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float_]
RenderFrame = NDArray[np.uint8]
Role = NewType("Role", str)


#######################################################
# Factories
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


def parallel_env(**kwargs):
    return MultInspect(**kwargs)


#######################################################
# Auxiliary Functions
#######################################################
def unit(v):
    if len(v.shape) == 1:
        v = v.reshape(1, -1)
    return v / np.linalg.norm(v, axis=1)[:, None]


def proj(v, onto):
    return onto * np.dot(v, onto) / np.dot(onto, onto)


def frame(r, v):
    r = unit(r)
    v = unit(v)
    n = np.cross(r, v, axis=1)

    return np.stack([r, v, n], axis=2)


def hills_frame(orb):
    r = orb.r
    v = orb.v
    v_p = v - proj(v, onto=r)
    return frame(r, v_p)[0].decompose().value


# def accel(self, t0, state, k, rate=1e-5):
#     """Constant acceleration aligned with the velocity."""
#     v_vec = state[3:]
#     # v_vec = state[:3]
#     norm_v = (v_vec * v_vec).sum() ** 0.5
#     return -rate * v_vec / norm_v

# def f(self, t0, u_, k):
#     # t_0: time to evaluate at
#     # u_ = [x,y,z,vx,vy,vz] in earth coords, rather annoyingly unitless.
#     # Assumed units (I've tested this) are km and km/s

#     # U.append(u_)
#     du_kep = func_twobody(t0, u_, k)
#     # ax, ay, az = self.accel(t0, u_, k, rate=1e-5)
#     ax, ay, az = self.acc
#     du_ad = np.array([0, 0, 0, ax, ay, az])
#     return du_kep + du_ad


#######################################################
# Class
#######################################################
class MultInspect(ParallelEnv):
    """
    We assuime six axis movement:
        action space: [X-Thrust, Y-Thrust, Z-Thrust,
                       X-Clockwise Torque, Y-Clockwise Torque, Z-Clockwise Torque]

    Some notes on linear algebra conventions and frames:

    All frames have columns as vectors in the background coordinate system R, so
    for any frame M, M * v maps v to R. To translate back, a vector in w in R is mapped to
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
        A list of all possible_agents the environment could generate. Equivalent to the list of agents in the
        observation and action spaces. This cannot be changed through play or resetting.
    max_num_agents: int
        The length of the possible_agents list.
    observation_spaces: dict[AgentID, gym.spaces.Space]
        A dict of the observation spaces of every agent, keyed by name. This cannot
        be changed through play or resetting.
    action_spaces: dict[AgentID, gym.spaces.Space]
        A dict of the action spaces of every agent, keyed by name. This cannot be changed through play or resetting.


    Methods
    -------------

    step(actions: dict[str, ActionType]) → tuple[dict[str, ObsType],
                                                 dict[str, float],
                                                 dict[str, bool],
                                                 dict[str, bool],
                                                 dict[str, dict]]
        Receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, terminated dictionary,
        truncated dictionary and info dictionary, where each dictionary is keyed by the agent.
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

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        "name": "MAInspect",
    }

    def __init__(self, render_mode="rgb_array", **kwargs) -> None:
        """ """
        self.render_mode = render_mode
        self.augment(params=kwargs)

    def augment(self, params: dict) -> None:
        self._setup_gym_env(env_params=params)

    def _setup_gym_env(self, env_params: dict) -> None:
        # Setup parameters. All of this may be overwritten by the
        # params dictionary if the appropreate key exists.
        # The list below provides the default parameters

        #######################
        # Parameters
        #######################
        self.six_axis: bool = True
        self.num_deputies: int = 1
        self.max_episode_steps: int = 100
        self.num_points: int = 20

        self.MAXTIME = 1 << u.h
        self._TIME_PER_STEP = 0.1 << u.min

        # Compatibility
        # This disable some agents terminating before others.
        self._SB_SAFTY_MODE = True

        # Game Mode:
        self._TRAIN_WAYPOINTER = False
        self._WAYPOINT_ARRIVAL_PROX = 10 << u.m
        self._WAYPOINT_ARRIVE_REWARD = 1
        self._WAYPOINT_NEAREST_REWARD = False
        self._WAYPOINT_NEAREST_REWARD_SCALE = 10
        self._WAYPOINT_START_AWAY_FROM_DEPUTY = True

        # Initial Conditions
        self.INIT_ALT = 400 << u.km
        self.INIT_ALT_OFFSET = 50 << u.m
        self.RAAN = 0 << u.deg  # Angle Around the Circular Orbit
        self.ARGLAT = 0 << u.deg  # Latitude updown
        self.INC = 0 << u.deg  # Direction of orbit
        self._STARTING_DISTANCE_FROM_CHIEF = 150 << u.m

        self.offset_angle = (1 / 10) * (
            self._STARTING_DISTANCE_FROM_CHIEF / self.INIT_ALT
        ).decompose().value << u.rad

        # Action Frame
        #  Options should be Hills, Orientation, Chief Hills
        self.ACTION_FRAME: str = "Chief Hills"
        self.OBSERVATION_FRAME: str = "Chief Hills"
        #  Options should be Background, ActionFrame
        self.ORI_CONSTANT_IN: str = "ActionFrame"

        self._OU_DIS = u.m
        self._OU_TIM = u.s
        self._OU_VEL = self._OU_DIS / self._OU_TIM

        self._CHIEF_PERIMETER: float = 50 << u.m
        self._DEPUTY_PERIMETER: float = 150 << u.m
        self._DEPUTY_MASS: float = 1 << u.kg
        self._DEPUTY_RADIUS: float = 1 << u.m

        self._SIM_OUTER_PARAMETER: float = 200 << u.m  #
        self._MAX_OUTER_PERIMETER: float = 300 << u.m  # If you leave here, you truncate
        self._STARTING_VEL_NORM: float = 0 << u.m / u.s  # 0.001

        self._TIME_PER_STEP: float = 30 << u.s
        self._DELTA_V_PER_THRUST: float = 0.01 << u.m / u.s
        self._DELTA_T: float = 90 << u.s
        self._TAU_per_THRUST: float = 0.1
        self._USE_ANGULAR_MOMENTUM = False

        self._OBS_REWARD: float = 0.02  # 20
        self._REWARD_FOR_SEEING_ALL_POINTS: float = 1  # 200
        self._CRASH_REWARD: float = 0
        self._REWARD_PER_STEP: float = 0.0  # -.1  # Reward per tick
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
        if hasattr(env_params, "args"):
            self.master_seed: int = env_params.args.master_seed
        else:
            self.master_seed: int = 487924

        if self._TRAIN_WAYPOINTER:
            self.num_points = self.num_deputies
            self._WAYPOINT_ARRIVAL_PROX = self._WAYPOINT_ARRIVAL_PROX << u.m

        # Check parameters from input
        assert self._MIN_VISION >= 0, "_MIN_VISION must be >= 0"
        assert self._MAX_VISION > self._MIN_VISION, "_MAX_VISION must be > _MIN_VISION"
        assert (self._VISION_ARC >= 0) and (
            self._VISION_ARC <= np.pi << u.rad
        ), "_VISION_ARC must be [0, pi]"

        ############
        # Setup things
        ############
        # Setup actual Gym Env based on the above
        self.agents: list[Role] = [
            Role(f"player_{i}") for i in range(self.num_deputies)
        ]
        self.possible_agents = self.agents

        # Agent Velocity, Other Dep. Rel Position and Vel, Point Rel Position
        self._VISION: dict[Role, list[float]] = {
            role: [self._VISION_ARC, self._MIN_VISION, self._MAX_VISION]
            for role in self.possible_agents
        }

        # observation spaces
        if self._TRAIN_WAYPOINTER:
            raw_low_values: list[float] = [
                # -np.ones(3) * self.INIT_ALT.to(self._OU_DIS).value,
                -self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                -self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                -2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                -2 * self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Single Waypoint,
            ]

            raw_high_values: list[float] = [
                # np.ones(3) * self.INIT_ALT.to(self._OU_DIS).value,
                self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Chief Position
                self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * self.num_deputies),  # Deputy velocity
                2
                * self._MAX_OUTER_PERIMETER.value
                * np.ones(3 * (self.num_deputies - 1)),  # Dep. Positions, except me
                2 * self._MAX_OUTER_PERIMETER.value * np.ones(3),  # Single Waypoint,
            ]
        else:
            raw_low_values: list[float] = [
                # -np.ones(3) * self.INIT_ALT.to(self._OU_DIS).value,
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
                # np.ones(3) * self.INIT_ALT.to(self._OU_DIS).value,
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

        # action spaces
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

        self.observation_spaces = {role: obs_space for role in self.possible_agents}
        self.action_spaces = {role: act_space for role in self.possible_agents}

        # positional embeddings
        self.ori = np.zeros(
            [self.num_deputies, 3, 3]
        )  # Depute Orientation relative to absolute earth frame
        self.rot = np.zeros([self.num_deputies, 3])  # Depute Angular Momentum

        self.pos = np.zeros([self.num_deputies, 3])  # Depute Position Holder
        self.vel = np.zeros([self.num_deputies, 3])  # Depute Velocity Holder

        self.chief_frame = np.zeros(
            [3, 3]
        )  # Current frame for the chief relative to absolute earth frame
        self.frames = np.zeros(
            [self.num_deputies, 3, 3]
        )  # Current frame for the deputies relative to absolute earth frame
        self.pts = np.zeros([self.num_deputies, 3])  # Chief Inspection Points
        self.nor = np.zeros([self.num_deputies, 3])  # Chief Inspection Normals

        # setup rest of variables
        self.reset()

    def _apply_parameters(self, params: dict) -> None:
        for k, v in params.items():
            self.__dict__[k] = copy.copy(v)

    def parallel_env(self, **kwargs):
        return self

    def seed(self, seed: int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed))

    def observation_space(self, agent: Role) -> gym.spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: Role) -> gym.spaces.Box:
        return self.action_spaces[agent]

    def close(self) -> None:
        pass

    def state(self) -> None:
        raise NotImplementedError("State Not Implemented.")

    # Reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[Any, Any]] = None,
        render_path: Optional[str] = None,
    ) -> tuple[dict[Role, Any], dict[Role, dict[str, Any]]]:
        """ """
        # reset simulation time
        self.sim_steps = 0
        self.unobserved_points = set(range(self.num_points))

        # Reset Seed
        local_seed = seed if (seed is not None) else self.master_seed
        self.seed(seed=local_seed)
        local_og = ortho_group(dim=3, seed=self.np_random)

        # Create Orbits for centers of mass
        self.orb = dict()

        # Construct Chief
        self.orb["chief"] = Orbit.circular(Earth, alt=self.INIT_ALT)

        # Construct Deputies
        self._active = np.full((self.num_deputies,), fill_value=True, dtype=bool)

        # Deputy Orbits
        for i, role in enumerate(self.possible_agents):
            self.orb[role] = Orbit.circular(
                Earth,
                alt=self.INIT_ALT + i * self.INIT_ALT_OFFSET,
                raan=self.offset_angle,
            )

        self.ori = np.stack([local_og.rvs() for i in range(self.num_deputies)])
        self.rot = self._random_sphere(self.num_deputies, radius=1)

        # Waypointer
        if self._TRAIN_WAYPOINTER:
            # Points are fixed in the chiefs hills frame, so we have to convert there:
            if self._WAYPOINT_START_AWAY_FROM_DEPUTY:
                # position of things in chief frame
                cframe = hills_frame(orb=self.orb["chief"])
                pos = self.pos @ cframe  # = (cframe.T @ self.pos.T).T

                self.pts = (
                    np.zeros([self.num_deputies, 3]) << self._DEPUTY_PERIMETER.unit
                )
                for i in range(self.num_deputies):
                    pt = self._random_sphere(1, radius=self._DEPUTY_PERIMETER)
                    while (
                        np.linalg.norm(pt - pos[i, :]) < 2 * self._WAYPOINT_ARRIVAL_PROX
                    ):
                        pt = self._random_sphere(1, radius=self._DEPUTY_PERIMETER)

                    self.pts[i] = pt
            else:
                self.pts = self._random_sphere(
                    self.num_points, radius=self._DEPUTY_PERIMETER
                )

            self.closest_distance = (
                2 * np.ones(self.num_deputies) * self._MAX_OUTER_PERIMETER
            )
        else:
            self.pts = self._random_sphere(
                self.num_points, radius=self._CHIEF_PERIMETER
            )

        self.chief_angular_momentum = None

        # Make Initial Observations

        # Compute rewards
        if self._TRAIN_WAYPOINTER:
            obs = self._make_waypoint_observations()
        else:
            obs = self._make_observations()

        self.truncated = {role: False for role in self.possible_agents}

        # Initial relative positions and velocities
        self.pos = np.stack(
            [self.orb[role].r - self.orb["chief"].r for role in self.possible_agents]
        )
        self.vel = np.stack(
            [self.orb[role].v - self.orb["chief"].v for role in self.possible_agents]
        )

        # Setup Reward Structure for waypoints
        self.pt_prox = []
        for i in range(self.num_points):
            self.pt_prox.append(
                multivariate_normal(mean=self.pts[i], cov=self._REW_COV_MTRX)
            )

        self.cum_rewards = np.zeros(self.num_deputies)

        # Setup Longterm Rendering
        self.render_path: Optional[str] = render_path
        self.frame_store: list[RenderFrame] = []
        if render_path:
            frame = self.render()
            assert frame is not None
            self.frame_store = [frame]

        self.render_data = dict()

        # Hold absolute orientations for telemetry:
        self.ori_background = []
        for player, role in enumerate(self.possible_agents):
            frame = self._local_frame(
                local_frame=self.ACTION_FRAME, role=role, role_count=player
            )

            # self.ori.shape = (num_deputies, [h,p,y], R)
            if self.ORI_CONSTANT_IN == "Background":
                self.ori_background.append(self.ori[player, :, :])
            else:
                assert self.ORI_CONSTANT_IN == "ActionFrame"
                self.ori_background.append(self.ori[player, :, :] @ frame)

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
        # logger.info("Orientation: %s", self.ori)
        # truncated = self.truncated
        self.terminated_this_step = False

        # Propagate Motion In Time
        self._propagate_objects(action)

        self.pos = np.stack(
            [self.orb[role].r - self.orb["chief"].r for role in self.possible_agents]
        )
        self.vel = np.stack(
            [self.orb[role].v - self.orb["chief"].v for role in self.possible_agents]
        )

        # Compute rewards
        if self._TRAIN_WAYPOINTER:
            self._compute_waypoint_reward()
            obs = self._make_waypoint_observations()
        else:
            # Compute Which Points Seen
            just_seen = self._detect_points()
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

        self.terminated = {
            agt: self.terminated_this_step for agt in self.possible_agents
        }

        # Remove Terminated and Truncated agents from available agent lists
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

    ###########################
    # Observation-related Functions
    ###########################
    def _local_frame(
        self, local_frame: str, role: str, role_count: Optional[int] = None
    ):
        if local_frame == "Hills":
            return hills_frame(self.orb[role])
        elif local_frame == "Chief Hills":
            return hills_frame(self.orb["chief"])
        else:
            assert local_frame == "Orientation"
            return self.ori[role_count]

    def _make_pv(self):
        # aux function for observations
        # agent positions and velocities
        pos_d = (
            np.stack([self.orb[role].r for role in self.possible_agents])
            .T.to(self._OU_DIS)
            .value
        )
        vel_d = (
            np.stack([self.orb[role].v for role in self.possible_agents])
            .T.to(self._OU_VEL)
            .value
        )

        # chief position and velocity
        pos_c = self.orb["chief"].r.reshape(-1, 1).to(self._OU_DIS).value
        vel_c = self.orb["chief"].v.reshape(-1, 1).to(self._OU_VEL).value

        # point positions
        pos_p = self.pts.T.to(self._OU_DIS).value

        return pos_d, vel_d, pos_c, vel_c, pos_p

    def _make_rel_pv(
        self, obs_frame: str, role: Role, role_count: int, vel_d, vel_c, pos_d, pos_c
    ):
        # aux function for player-frame positions and velocities
        #  these are every other player/point relative to me

        # Recall: Moving from absolute coords into a frame is left multiplacation
        # by the transpose

        # My frame
        frame = self._local_frame(
            local_frame=obs_frame, role=role, role_count=role_count
        )

        # Velocities
        # Relative velocities of all other deputies in my frame
        vel_in_frame = frame.T @ (vel_d - vel_d[:, [role_count]])
        dep_rel_v = np.delete(arr=vel_in_frame, obj=role_count, axis=1)

        # Relative velocity of chief in my frame
        chief_rel_v = frame.T @ (vel_c - vel_d[:, [role_count]])

        # Positions
        # Relative positions of all other deputies in my frame
        dep_rel_p = frame.T @ np.delete(
            arr=pos_d - pos_d[:, [role_count]], obj=role_count, axis=1
        )

        # Relative position of chief in my frame
        chief_rel_d = pos_c - pos_d[:, [role_count]]
        chief_rel_p = frame.T @ chief_rel_d

        return frame, chief_rel_d, chief_rel_p, chief_rel_v, dep_rel_p, dep_rel_v

    def _make_waypoint_observations(
        self, obs_frame: Optional[str] = None
    ) -> dict[Role, FloatArray]:
        """ """
        # Return dictionary
        # Play role will be keys, observations will be values
        obs = dict()

        if obs_frame is None:
            obs_frame = self.OBSERVATION_FRAME

        # get positions and velocities
        pos_d, vel_d, pos_c, vel_c, pos_p = self._make_pv()

        # loop over all deputies
        for i, role in enumerate(self.possible_agents):
            # Relative positions and velocities
            (
                frame,
                chief_rel_d,
                chief_rel_p,
                chief_rel_v,
                dep_rel_p,
                dep_rel_v,
            ) = self._make_rel_pv(
                obs_frame := obs_frame,
                role=role,
                role_count=i,
                vel_d=vel_d,
                vel_c=vel_c,
                pos_d=pos_d,
                pos_c=pos_c,
            )

            # Relative positions of all points in my frame
            #  this is the main difference from the other observations.
            #  We're only looking at the current waypoint in this version.
            rel_pts = frame.T @ (pos_p[:, [i]] + chief_rel_d)

            # Put the observation space together
            tmp = np.concatenate(
                [
                    # pos_d[:, [i]],
                    chief_rel_p,
                    chief_rel_v,
                    dep_rel_v,
                    dep_rel_p,
                    rel_pts,
                ],
                axis=1,
            ).T
            # print(tmp)

            # Store observation space under appropriate role
            obs[role] = tmp.reshape(-1)

        return obs

    def _make_observations(
        self, obs_frame: Optional[str] = None
    ) -> dict[Role, FloatArray]:
        """ """
        # Return dictionary
        # Play role will be keys, observations will be values
        obs = dict()

        if obs_frame is None:
            obs_frame = self.OBSERVATION_FRAME

        # Setup point mask
        #  hide points that have been observed
        observedPoints = list(
            set(range(self.num_points)).difference(self.unobserved_points)
        )
        pt_mask = np.ones(self.num_points)
        pt_mask[observedPoints] = 0

        # get positions and velocities
        pos_d, vel_d, pos_c, vel_c, pos_p = self._make_pv()

        # loop over all deputies
        for i, role in enumerate(self.possible_agents):
            # Relative positions and velocities
            (
                frame,
                chief_rel_d,
                chief_rel_p,
                chief_rel_v,
                dep_rel_p,
                dep_rel_v,
            ) = self._make_rel_pv(
                obs_frame := obs_frame,
                role=role,
                role_count=i,
                vel_d=vel_d,
                vel_c=vel_c,
                pos_d=pos_d,
                pos_c=pos_c,
            )

            # Relative positions of all points in my frame
            #  This is the main difference from the waypointer - here, we can see
            #  all the points, while the waypointer is only looking at its next
            #  waypoint.
            rel_pts = frame.T @ (pos_p + chief_rel_d)

            # Put the observation space together
            tmp = np.concatenate(
                [
                    # pos_d[:, [i]],
                    chief_rel_p,
                    chief_rel_v,
                    dep_rel_v,
                    dep_rel_p,
                    rel_pts,
                ],
                axis=1,
            )

            # Store observation space under appropriate role
            obs[role] = np.concatenate([tmp.reshape(-1), pt_mask])

        return obs

    ###########################
    # Reward-related Functions
    ###########################
    def _compute_reward(self, just_seen: list[list[int]]) -> float:
        # setup holder object for this step
        self.sra = np.zeros(self.num_deputies)  # Step Reward Array
        self.step_rewards = {
            role: self.sra[i : i + 1] for i, role in enumerate(self.possible_agents)
        }  # Slices are pointers to the reward array

        # shared rewards
        self._game_length_exceed()
        self._prox_rewards()

        # inspect-only rewards
        self._inspect_reward(just_seen)
        self._seeing_all_points()

    def _compute_waypoint_reward(self) -> float:
        # setup holder object for this step
        self.sra = np.zeros(self.num_deputies)  # Step Reward Array
        self.step_rewards = {
            role: self.sra[i : i + 1] for i, role in enumerate(self.possible_agents)
        }  # Slices are pointers to the reward array

        # shared rewards
        self._game_length_exceed()
        self._prox_rewards()

        # waypoint-only rewards
        self._go_towards_waypoint()

    def _game_length_exceed(self):
        # increment number of steps sim has taken
        self.sim_steps += 1

        # check if done
        if self.sim_steps > self.max_episode_steps:
            # Technically we were truncated
            self.truncated = {role: True for role in self.possible_agents}
            # Set sim to finished
            self.terminated_this_step = True

    def _prox_rewards(self):
        # get distance for each player
        player_dist = np.linalg.norm(self.pos, axis=1)

        # Active and left outer perimeter?
        out_perim = np.logical_and(
            self._active, player_dist > self._MAX_OUTER_PERIMETER
        )

        # Active and outside the safe zone?
        out_safe = np.logical_and(self._active, player_dist > self._SIM_OUTER_PARAMETER)

        # Active and inside the chief?
        in_chief = np.logical_and(self._active, player_dist < self._CHIEF_PERIMETER)

        # Adjust rewards
        self.sra[out_perim] += self._REWARD_FOR_LEAVING_OUTER_PERIMETER
        self.sra[out_safe] += self._REWARD_FOR_LEAVING_PARAMETER
        self.sra[in_chief] += self._CRASH_REWARD

        # Adjust active agents
        #  self.terminated is a dict, which needs the keys from self.possible_agents.
        #  self.possible_agents is a list, which can't use boolean indexing.
        done_idx = np.where(np.logical_or(out_perim, in_chief))[0]
        for i in done_idx:
            self._active[i] = False
            self.terminated[self.possible_agents[i]] = True

        # Check active agents
        #  SB3 compatibility requires us end the sim if _any_ agent is done
        if not np.all(self._active):
            self.terminated_this_step = True

    def _inspect_reward(self, just_seen):
        """ """
        # Object to track newly-seen points
        #  This allows 2 satellites to see the same point in the same step.
        track_pts: set = set()

        for i, role in enumerate(self.possible_agents):
            # only work on active agents
            if self._active[i]:
                # Initialize reward
                #  Start with the survival reward
                reward: float = self._REWARD_PER_STEP

                # Check if seen points are new
                new_pts: set = self.unobserved_points.intersection(just_seen[i])

                # Check if new points were seen
                if new_pts:
                    # Reward for seeing points for the first time
                    reward += len(new_pts) * self._OBS_REWARD
                    # Track newly-seen points
                    track_pts.update(new_pts)

                # update agent reward
                self.step_rewards[role] += reward

        # update unseen points
        self.unobserved_points = self.unobserved_points.difference(track_pts)

    def _seeing_all_points(self):
        # check if we have seen all the points
        if len(self.unobserved_points) == 0:
            # fraction of simtime left
            R = (self.max_episode_steps - self.sim_steps) / self.max_episode_steps
            # all agents get reward for finishing task
            self.sra += R * self._REWARD_FOR_SEEING_ALL_POINTS
            # stop the sim this step
            self.terminated_this_step = True

    def _go_towards_waypoint(self):
        # We're going to change the reward structure to include waypointing

        # Points are fixed in the chiefs hills frame, so we have to convert there:
        cframe = hills_frame(orb=self.orb["chief"])
        pos = self.pos @ cframe  # = (cframe.T @ self.pos.T).T

        for i, role in enumerate(self.possible_agents):
            # only work on active agents
            if self._active[i]:
                # We get some reward for going towards a point
                self.step_rewards[role] += (
                    self.pt_prox[i].pdf(pos[i, :]) * self._PROX_RWD_SCALE
                )

                # calculate distance to thing - JB, idk what this is doing
                pt_dist = np.linalg.norm(pos[i, :] - self.pts[i])

                # replace shortest distance
                if self.closest_distance[i] > pt_dist:
                    self.closest_distance[i] = pt_dist

                # did we arrive at a waypoint?
                if pt_dist < self._WAYPOINT_ARRIVAL_PROX:
                    # update my reward for arriving!
                    self.step_rewards[role] += self._WAYPOINT_ARRIVE_REWARD
                    # end simulation
                    #  we assume there's only 1 deputy, sim ends when you get your point
                    self.terminated_this_step = True

        # TODO: JB - idk what this is doing either
        if self._WAYPOINT_NEAREST_REWARD and self.terminated_this_step:
            # reward everyone based on where they are
            self.sra += (
                self._WAYPOINT_NEAREST_REWARD_SCALE / self.closest_distance.value
            )

    def _detect_points(self):
        # List to return
        #  We will fill this list with sublists of points each agent saw
        just_seen = []

        # position of all agents in chief hills frame
        cframe = hills_frame(orb=self.orb["chief"])
        pos = self.pos @ cframe

        for player, role in enumerate(self.possible_agents):
            # point holder
            seen_points: list[int] = []

            # only work on active agents
            if self._active[player]:
                # Detect Points:
                #  This handles occlusion by the chief
                #  Basically, what points are on the same side of the chief as you
                seeable_points = np.where(
                    (self.pts * (pos[player, :] - self.pts)).sum(axis=1) > 0
                )[0]

                ##########
                # Spherical Vision
                #########
                # loop over possible points
                for point in seeable_points:
                    # Cone References
                    # https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space
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
                    # logger.info("Seen Points: %s %s", self.pts[point, :], pos)
                    v = self.pts[point, :] - pos[player, :]  # In chiefs hills frame

                    # Distance relative to agent
                    #  this is the same as the distance in the deputy frame, but
                    #  doesn't require the transformation
                    p_dist = np.linalg.norm(v)

                    # logger.info("p_dist: %s", p_dist)

                    # Check if distance is within shell
                    #  Ignoring angles at the moment
                    #  Less than max, more than min
                    #   I assume we're usually too far away
                    if (p_dist < self._VISION[role][2]) and (
                        p_dist >= self._VISION[role][1]
                    ):
                        # Orient point in deputy frame
                        # logger.info("pdf: %s %s", self.ori[player].T, v.T)
                        if self.ORI_CONSTANT_IN == "Background":
                            d_ori = (
                                self.ori[player] @ cframe
                            )  # [[h,p,y], R] [R, v] = [[h,p,y], v]
                        else:
                            assert self.ORI_CONSTANT_IN == "ActionFrame"

                            frame = self._local_frame(
                                local_frame=self.ACTION_FRAME,
                                role=role,
                                role_count=player,
                            )
                            d_ori = (
                                self.ori[player] @ frame.T @ cframe
                            )  # [[h,p,y], A] [A, R] [R, v] = [[h,p,y], v]

                        pdf = d_ori @ v.T  # vector from deputy to point in ori frame.

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

            # update seen points
            #  this will append an empty list if an agent is not active
            just_seen.append(seen_points)

        return just_seen

    ###########################
    # Reset-related Functions
    ###########################
    def _random_sphere(self, n_points: int, radius: float) -> FloatArray:
        """ """
        # Generate random points on a sphere
        pts: FloatArray = -1 + 2 * self.np_random.normal(size=[n_points, 3])
        pts = pts / np.linalg.norm(pts, axis=1).reshape(-1, 1)
        pts = pts * radius

        return pts

    ###########################
    # Step-related Functions
    ###########################
    # Physics goes here
    def _propagate_objects(self, actions):
        # Poliastro assumes earth centered coordinates. It fixes an x,y, and z. So
        # vectors in poliastro objects are in terms of of an absolute coordinate system.

        # Translate actions into action frame and apply impulses
        self.ori_background = []

        for player, (role, act) in enumerate(actions.items()):
            # Compute Velocity Change
            dv = act[:3].reshape(-1, 1) * self._DELTA_V_PER_THRUST
            frame = self._local_frame(
                local_frame=self.ACTION_FRAME, role=role, role_count=player
            )

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

            # self.ori.shape = (num_deputies, [h,p,y], R)
            if self.ORI_CONSTANT_IN == "Background":
                T = (
                    frame @ R.as_matrix() @ frame.T
                )  # x columns to frame, rotate, then back to x
                self.ori[player, :, :] = (T @ self.ori[player, :, :].T).T

                self.ori_background.append(self.ori[player, :, :])
            else:
                assert self.ORI_CONSTANT_IN == "ActionFrame"
                # logger.info("Incoming Rotation: %s %s", R, self.ori[player, :, :])
                self.ori[player, :, :] = (R.as_matrix() @ self.ori[player, :, :].T).T

                self.ori_background.append(self.ori[player, :, :] @ frame.T)

        # logger.info("Orientation: %s", self.ori)

        # Rotate Chief Points:
        if self.chief_angular_momentum is not None:
            R = self.chief_angular_momentum
            self.pts = R @ self.pts

        # Propagate Orbits
        for role, orb in self.orb.items():
            self.orb[role] = orb.propagate(self._TIME_PER_STEP)

    # Visualizations
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
        fig = plt.figure(figsize=[15, 10])
        canvas = FigureCanvas(fig)

        poss = (
            np.stack(
                [
                    self.orb[role].r - self.orb["chief"].r
                    for role in self.possible_agents
                ]
            )
            .T.to(self._OU_DIS)
            .value
        )
        vels = (
            np.stack(
                [
                    self.orb[role].v - self.orb["chief"].v
                    for role in self.possible_agents
                ]
            )
            .T.to(self._OU_VEL)
            .value
        )

        cframe = hills_frame(self.orb["chief"])

        chief_rel_poss = (cframe.T @ poss).T
        ##########
        # Title
        ##########
        # plot scores as title
        # print(chief_rel_poss.shape)

        score_title = "    ".join(
            [
                f"Player {i}: {self.cum_rewards[i]:.3f} {np.linalg.norm(chief_rel_poss[i] - self.pts[i].to(self._OU_DIS).value):.3f}"  # noqa: E501
                for i in range(self.num_deputies)
            ]
        )
        fig.suptitle(
            t=score_title, horizontalalignment="center", verticalalignment="center"
        )

        ##########
        # Setup Plot Area
        ##########
        # Plot just the chief, or the chief and one agent-centric view?
        if vision is not None:
            ax = fig.add_subplot(231, projection="3d")
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
            x, y, z = poss[:, [i]]
            ax.plot3D(x, y, z, marker="^", linewidth=0, label=f"player_{i}")

        # Velocity
        # Plot line collection: LC = np.array([N,S,D]), num lines, num of points
        # per line, dim of space, so LC[2,0,:] is point 0 on line 2

        TPS = (self._TIME_PER_STEP * self._OU_VEL / self._OU_DIS).decompose().value

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
            s = (
                np.linalg.norm(unobserved_pos.to(self._OU_DIS).value + view_vec, axis=1)
                - 40
            )
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
            s = (
                np.linalg.norm(observed_pos.to(self._OU_DIS).value + view_vec, axis=1)
                - 40
            )
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
        for i, s in enumerate(["x", "y", "z"]):
            lim = getattr(ax, f"get_{s}lim")()
            lim = [
                min(
                    max(mins[i], -300),
                    -self._CHIEF_PERIMETER.to(self._OU_DIS).value + 10,
                ),
                max(
                    min(maxes[i], 300),
                    self._CHIEF_PERIMETER.to(self._OU_DIS).value + 10,
                ),
            ]
            widths.append(lim[1] - lim[0])
            centers.append((lim[1] + lim[0]) / 2)

        w = max(widths) / 2
        for i, s in enumerate(["x", "y", "z"]):
            getattr(ax, f"set_{s}lim")([centers[i] - w, centers[i] + w])

        ##########
        # Render Agent View
        ##########
        if vision is not None:
            ax2 = fig.add_subplot(232, projection="3d")
            all_obs = self._make_observations(obs_frame="Orientation")
            role = cast(Role, f"player_{vision}")
            obs = all_obs[role]

            mask = obs[-self.num_points :]
            obs = obs[: -self.num_points].reshape(3, -1)
            x = obs[:, -self.num_points :]

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

        ax3 = fig.add_subplot(233)

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
        ax4 = fig.add_subplot(234)

        if "frames" not in self.render_data.keys():
            self.render_data["frames"] = []

        if "chief_rel_poss" not in self.render_data.keys():
            self.render_data["chief_rel_poss"] = []

        c_idx = list(self.orb.keys()).index("chief")
        frame = hills_frame(self.orb["chief"])

        chief_rel_poss = (frame.T @ poss.T).T
        # print(chief_rel_poss[c_idx,:])
        chief_rel_poss = chief_rel_poss - chief_rel_poss[c_idx, :]

        self.render_data["chief_rel_poss"].append(chief_rel_poss)
        self.render_data["frames"].append(frame)

        hills = np.array(self.render_data["chief_rel_poss"])

        for i in range(N):
            ax4.plot(hills[:, i, 1], hills[:, i, 0])

        ax4.scatter(self.pts[:, 1], self.pts[:, 0])

        chief = plt.Circle(
            (0, 0), self._CHIEF_PERIMETER.to(self._OU_DIS).value, color="k", fill=False
        )
        ax4.add_patch(chief)
        ax4.set_title("Chief Centered Hill Frame")
        ax4.set_xlabel("Mometum Direction")
        ax4.set_ylabel("Altitude")

        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()

        if xlim[1] - xlim[0] > ylim[1] - ylim[0]:
            dx = xlim[1] - xlim[0]
            ylim = [np.mean(ylim) - dx / 2, np.mean(ylim) + dx / 2]
        else:
            dy = ylim[1] - ylim[0]
            xlim = [np.mean(xlim) - dy / 2, np.mean(xlim) + dy / 2]

        ax4.set_xlim(xlim)
        ax4.set_ylim(xlim)

        ax5 = fig.add_subplot(235)

        for i in range(N):
            ax5.plot(hills[:, i, 2], hills[:, i, 0])

        ax5.scatter(self.pts[:, 2], self.pts[:, 0])

        chief = plt.Circle(
            (0, 0), self._CHIEF_PERIMETER.to(self._OU_DIS).value, color="k", fill=False
        )
        ax5.add_patch(chief)
        ax5.set_title("Chief Centered Hill Frame (Chase)")
        ax5.set_xlabel("Off Axis")
        ax5.set_ylabel("Altitude")

        xlim = ax5.get_xlim()
        ylim = ax5.get_ylim()

        if xlim[1] - xlim[0] > ylim[1] - ylim[0]:
            dx = xlim[1] - xlim[0]
            ylim = [np.mean(ylim) - dx / 2, np.mean(ylim) + dx / 2]
        else:
            dy = ylim[1] - ylim[0]
            xlim = [np.mean(xlim) - dy / 2, np.mean(xlim) + dy / 2]

        ax5.set_xlim(xlim)
        ax5.set_ylim(xlim)

        ax6 = fig.add_subplot(236)

        for i in range(N):
            ax6.plot(hills[:, i, 1], hills[:, i, 2])

        ax6.scatter(self.pts[:, 1], self.pts[:, 2])

        chief = plt.Circle(
            (0, 0), self._CHIEF_PERIMETER.to(self._OU_DIS).value, color="k", fill=False
        )
        ax6.add_patch(chief)
        ax6.set_title("Chief Centered Hill Frame (Earth Facing)")
        ax6.set_xlabel("Mometum Direction")
        ax6.set_ylabel("Off Axis")

        xlim = ax6.get_xlim()
        ylim = ax6.get_ylim()

        if xlim[1] - xlim[0] > ylim[1] - ylim[0]:
            dx = xlim[1] - xlim[0]
            ylim = [np.mean(ylim) - dx / 2, np.mean(ylim) + dx / 2]
        else:
            dy = ylim[1] - ylim[0]
            xlim = [np.mean(xlim) - dy / 2, np.mean(xlim) + dy / 2]

        ax6.set_xlim(xlim)
        ax6.set_ylim(xlim)

        # Draw
        if render_mode == "rgb_array":
            canvas.draw()

            # return figure as array of data
            #  rgba returns alpha channel - drop it
            data = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]

            # close plot
            plt.close(fig)

            return data

        plt.close(fig)
        return None


if __name__ == "__main__":
    from PIL import Image
    from tqdm import tqdm

    env = MultInspect(
        num_deputies=4, _SIM_OUTER_PARAMETER=300 << u.m, _CHIEF_PERIMETER=50 << u.m
    )
    env.seed(0)
    env.reset()

    # env.render()

    frames = []
    for i in tqdm(range(200)):
        if i <1:
            act = {role: np.array([10,0,0,0,0,0]) for role in env.possible_agents}
        else:
            act = {role: np.array([0,0,0,0,0,0]) for role in env.possible_agents}
    
        # act = {role: np.array([0,0,0,0,0,0]) for role in env.possible_agents}
        # act = {role: 0.5 - np.random.random(6) for role in env.possible_agents}
        (
            obs,
            _,
            _,
            _,
            _,
        ) = env.step(act)
        # print(obs["player_0"][:3])
        frames.append(env.render())

    Image.fromarray(frames[0])

    imgs = [Image.fromarray(frame) for frame in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    model_dir = ""
    imgs[0].save(
        model_dir + f"_eval_run_{i}.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=500,
        loop=0,
    )


# %%
# from PIL import Image
# from IPython.display import display

# # %%
# display(Image.fromarray(env.render()))
# print("Dist:", np.linalg.norm(env.pts - env.pos))
# env.orb["player_0"].r - env.orb["chief"].r

# env.step({"player_0":np.array([0,0,0])})

# obs, rew, term, trunc, info = env.step({"player_0":np.array([1,0,0])})
# # %%
# obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,0])})
# display(Image.fromarray(env.render()))
# # %%
# env.reset()
# print(env.pts)
# obs, rew, term, trunc, info = env.step({"player_0":np.array([1,0,0])})
# env.render()
# obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,0])})
# env.render()
# obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,0])})
# env.render()
# obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,0])})
# env.render()
# obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,-1])})
# env.render()
# for _ in range(20):
#     obs, rew, term, trunc, info = env.step({"player_0":np.array([0,0,0])})
#     env.render()
# display(Image.fromarray(env.render()))
# %%
