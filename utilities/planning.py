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

from .timelines import Timeline, Timelines, TimelineEvent
import numpy as np
from numpy.random import Generator, PCG64DXSM
import copy
import pandas as pd

class State:
    def __init__(self, parameters, seed, trajectory, current_t=0):
        self.parameters = parameters
        self.seed = seed
        self.trajectory = trajectory
        self.current_t = current_t
        self.cummulative_rews = dict()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"parameters: {self.parameters}, seed: {self.seed}, trajectory len: {len(self.trajectory)}"


class Trajectory:
    def __init__(self, env, initial_return, intial_frame=None):
        self.players = copy.copy(env.possible_agents)
        self.trajectory = []
        self.actions = []
        self.step_returns = []
        self.agent_info = []
        self.add(
            env, None, initial_return, None
        )
        self.frames = []
        if intial_frame is not None:
            self.frames.append(intial_frame)

    def add(self, env, action, step_return, agent_info, frame=None):
        self.trajectory.append(
            {
                "action": action, 
                "step_return": step_return, 
                "agent_info": agent_info
             }
        )
        self.actions.append(action)
        self.step_returns.append(step_return)
        self.agent_info.append(agent_info)

        if frame is not None:
            self.frames.append(frame)

    def __getitem__(self, key):
        return self.trajectory[key]

    def __len__(self):
        return len(self.trajectory)
    

## Telemetry Object:
class Telemetry:
    def __init__(self, name, **kwargs):
        self.data = None
        self.data_labels = None
        self.xscale = None
        self.title = None
        self.name = None
        self.xlabel = None
        self.ylabel = None
        self.colors = None

        for k,v in kwargs.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v
            else:
                raise Exception(f"Keyword argument {k} not in telemetry class.")

    def __str__(self) -> str:
        shape = getattr(self.data, "shape", None)
        return f"Telemetry - name: {self.name}, shape: {shape}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def set_data(self, data):
        self.data = data
        self.data_labels = [""] * data.shape[0]
        self.colors = [None] * data.shape[0]

    def as_df(self):
        # print("Telemtry:", self.data, self.data_labels, self.xscale)

        tmp = pd.DataFrame(self.data.T)

        # print("Telemtry:", tmp)
        columns = copy.copy(self.data_labels)
        for i in range(len(columns)):
            if columns[i] is None:
                if self.name is not None:
                    columns[i] = self.name
                else:
                    columns[i] = ""

        tmp.columns = columns

        # print("Telemtry:", tmp)
        tmp.index = self.xscale
        return tmp

# Returns a fixed seed and fixed set of parameters
class BaseParameterGenerator:
    def __init__(self, seed, param):
        self.seed = seed
        self.param = param

    def sample(self):
        return (self.seed, self.param)

    def setseed(self, seed):
        pass

    def update_param(self, param):
        self.param = param

    def freeze(self):
        pass

    def thaw(self):
        pass

# This returns a random seed but a fixed set of parameters
class SeededParameterGenerator:
    def __init__(self, seed, param):
        self.np_random = Generator(PCG64DXSM(seed=seed))
        self.param = param
        self._frozen = False

    def sample(self):
        if self._frozen:
            return (self._frozen_random, self.param)
        else:
            return (self.np_random.integers(low=2**32), self.param)

    def setseed(self, seed):
        self.np_random = Generator(PCG64DXSM(seed=seed))

    def update_param(self, param):
        self.param = param

    def freeze(self):
        self._frozen = True
        self._frozen_random = self.np_random.integers(low=2**32)

    def thaw(self):
        self._frozen = False

# %%
class CommunicationSchedule(Timelines):
    @staticmethod
    def repeating(
        checkin_frequency=1, 
        checkin_offset=0, 
        blackout_frequency=1,
        blackout_length=0, 
        blackout_offset=0,
        allow_agent_break=True,
    ):
        ## This is going to secretely overload the standard timeline behavior to make it
        ## infinite
        tmp = CommunicationSchedule(length=1, allow_agent_break=allow_agent_break)
        tmp.__dict__["checkin_frequency"] = checkin_frequency
        tmp.__dict__["checkin_offset"] = checkin_offset
        tmp.__dict__["blackout_frequency"] = blackout_frequency
        tmp.__dict__["blackout_length"] = blackout_length
        tmp.__dict__["blackout_offset"] = blackout_offset
        tmp.__dict__["allow_agent_break"] = allow_agent_break,
        tmp.__dict__["repeating"] = True

        blackout_event = TimelineEvent(parameters=None, label="blackouts")
        checkin_event = TimelineEvent(parameters=None, label="checkins")
        
        def next_event(time):
            t1 = (checkin_frequency - time + checkin_offset) % checkin_frequency
            if t1 == 0:
                t1 = checkin_frequency
            return t1 + time, checkin_event
        
        tmp.checkins.__dict__["next_event"] = next_event

        def get(time, role):
            if ( (time - blackout_offset)  % blackout_frequency) < blackout_length:
                return True
            
            return False

        tmp.blackouts.__dict__["get"] = get

        return tmp

    
    @staticmethod
    def from_lists(length, blackouts = [], checkins = [], allow_agent_break=True):
        tmp = CommunicationSchedule(length, allow_agent_break)
        blackout_event = TimelineEvent(parameters=None, label="blackouts")
        checkin_event = TimelineEvent(parameters=None, label="checkins")
        
        for t in blackouts:
            tmp.blackouts.add_events(t, blackout_event)

        for t in checkins:
            tmp.checkins.add_events(t, checkin_event)

        return tmp

    def __init__(self, length, allow_agent_break=True):
        super().__init__(labels=["blackouts", "checkins"], length=length)
        self.ALLOW_AGENT_BREAK = allow_agent_break
        self.repeating = False

        def get(time, role):
            return False

        self.blackouts.__dict__["get"] = get
        # self.checkins.add_event(time=0, event=TimelineEvent(parameters=None, label="checkin"))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


    def __deepcopy__(self, memo):
        if self.repeating:
            return CommunicationSchedule.repeating(
                checkin_frequency=copy.copy(self.checkin_frequency), 
                checkin_offset=copy.copy(self.checkin_offset), 
                blackout_frequency=copy.copy(self.blackout_frequency),
                blackout_length=copy.copy(self.blackout_length), 
                blackout_offset=copy.copy(self.blackout_offset),
                allow_agent_break=copy.copy(self.allow_agent_break),
            )
        
        tmp = self.__class__(labels=[])
        tmp.timelines = copy.deepcopy(self.timelines)

        return(tmp)

class COA(Timeline):
    def to_dict(self):
        d = {
            time: {event.label: event.to_dict() for event in events} for time, events in self.timeline.items()
            }
        return d

    def get_actions(self, t):
        return self.timeline.get(t, default=None)