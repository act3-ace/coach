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

import sys, inspect

from utilities.planning import COA, TimelineEvent
from coach import CommunicationSchedule
import plotly.express as px 

from argparse import Namespace
from utilities.iotools import NumpyDecoder
import json
import os
import copy
import numpy as np
from itertools import product
import logging
logger = logging.getLogger(__name__)

from coach import (
    SeededParameterGenerator,
    BaseParameterGenerator,
)

import agents as agents_module
import directors as directors_module


def role_type(role):
    return role.split("_")[0]

class COACHIntegration:
    def __init__(self, 
        env_creator, 
        COACHEnvClass,
        parameters, 
        agents_module=agents_module
    ) -> None:
        self.agents_module=agents_module
        self.env_creator = env_creator
        self.COACHEnvClass = COACHEnvClass
        self.parameters = parameters
        self.render_gif = False
        
        self.library = dict()
        self._id_to_interface = dict()
        self._interface_to_id = dict()

        self.setup()

        self.plans = PlanLibrary(
            self,
            default_interfaces=self.default_interfaces,
            default_models=self.default_models,
        )

        self.actions_container = {}

    def get_current_models(self):
        return {role: self.get_interface_models(interface) for role, interface in self.plans.current.interfaces.items()}

    # def get_interfaces(self):
    #     return list(self.interfaces.keys())
    
    def set_current_interface(self, role, interface):
        self.current_interface[role] = interface

    def get_interfaces_by_role(self, role):
        return list(self.interfaces_by_role[role].keys())

    def get_current_interface(self, role):
        return self.current_interface[role]
    
    def get_interface_models(self, interface):
        return list(self.model_params_by_interface[interface].keys())
        
    def get_current_model(self):
        return self.current_model

    def set_current_model(self, role, model):
        self.current_model[role] = model

    def actions(self, role):
        agt = self.coach_env.agents[role]
        return agt.interface.action_dictionary

    # Deal with Plan Library
    def new_plan(self, name = None):
        self.plans.new_plan(name)
    
    def get(self, id):
        return(self.library[id])

    def keys(self):
        return(self.library.keys())
    
    def items(self):
        return(self.library.items())
    
    def values(self):
        return(self.library.values())
    
    def by_interface(self, interface):
        return(self._interface_to_id[interface])
    
    def get_by_interface(self, interface, id):
        return(self._interface_to_id[interface][id])

    def plot_COA(self, COA):

        logger.debug("coach_integration.plot_COA - Generated COA Trajectory: %s", COA)
        traj = self.coach_env.get_traj_from_coas({"player_0": COA})

        logger.debug("coach_integration.plot_COA - Generated COA Trajectory: %s", len(traj))

        return plot_trajectory_component(traj, {"player_0": COA}, self.coach_env.env)
    

    def setup(self):
        ## This should set up the game, get the information about it, and fix things like the 
        ## which roles can take which models. 
        ## COA_Env has agents, need to extract them from env. 
        # Create COA Env

        self.COACH_params = COACH_params = self.parameters["COACH_params"]
        self.env_params = env_params = self.parameters["env_params"]
        self.actor_params = actor_params = self.parameters["actor_params"]

        # Set up parameter generator
        if COACH_params["stochastic"]:
            self.parameter_generator = SeededParameterGenerator(23423, env_params)
        else:
            self.parameter_generator = BaseParameterGenerator(23423, env_params)

        # Set up comm schedule
        if "FIXED_STEPS_PER_COM" in COACH_params.keys():
            schedule_param = COACH_params["FIXED_STEPS_PER_COM"]
            self.comm_schedule = CommunicationSchedule.repeating(**schedule_param)
        else:  
            logger.info("Communication Schedule Is Non-repeating")
            self.comm_schedule = CommunicationSchedule(length=0)
        
        self.ACTION_PADDING = COACH_params.get("ACTION_PADDING", 0)
        self.MIN_NEXT_ACTION_TIME = COACH_params.get("MIN_NEXT_ACTION_TIME", 1)
        self.MAX_NEXT_ACTION_TIME = COACH_params.get("MAX_NEXT_ACTION_TIME", np.inf)

        self.coach_env = self.COACHEnvClass(
            env_creator=self.env_creator,
            parameter_generator=self.parameter_generator,
            # agents=agent_dict,
            fill_random=False,
            comm_schedule=self.comm_schedule,
            seed=COACH_params["seed"],
        )

        self.coach_env.augment(self.parameter_generator)
        self.coach_env.reset()

        self.roles = self.coach_env.env.agents
        self.role_to_idx = {role:idx for idx, role in enumerate(self.roles)}
        self.idx_to_role = {idx:role for idx, role in enumerate(self.roles)}
        
        self.role_types = list(set([role_type(role) for role in self.roles]))
        self.possible_models = self.coach_env.possible_models()


        # Setup Actor Interfaces
        self.interfaces_by_role = {role: dict() for role in self.roles}
        self.roles_by_interface = dict()
        self.interfaces = dict()
        self.model_params_by_interface = dict()
        self.model_params = dict()
        self.model_classes = dict()
        self.model_to_interface = dict()
        
        for if_name, if_params in self.actor_params["interfaces"].items():
            # Get the interface class
            if_Class = self.agents_module.__dict__.get(if_params["interface_class"])
            logger.debug("Current Class: %s", if_Class)
            logger.debug("Current Params: %s", if_params)
            # if_Class = self.coach_env.__class__.agent_selection.Interfaces[if_params["interface_class"]]
            
            # Get a reference interface, this is make sure that any model class instiated has the correct interface
            role = if_params.get("roles", self.roles)[0] # Get one applicable role
            if_reference = if_Class(role, self.coach_env.env, **if_params.get("iterface_parameters", dict()))
            logger.debug("Current Class: %s", if_reference)

            # Set up references to interfaces
            self.interfaces[if_name] = if_reference

            # Get roles for which the interface is valid
            for role in if_params.get("roles", self.roles):
                self.interfaces_by_role[role][if_name] = if_reference
                if if_name not in self.roles_by_interface.keys():
                    self.roles_by_interface[if_name] = []
                self.roles_by_interface[if_name].append(role)

            # Setup reference to model creation information
            self.model_params_by_interface[if_name] = if_params["models"]

            self.model_classes[if_name] = dict()
            for model_name, model_params in if_params["models"].items():
                self.model_to_interface[model_name] = if_name
                self.model_classes[if_name][model_name] = self.agents_module.__dict__.get(model_params["class_name"]) 
                self.model_params[model_name] = model_params

        # Setup Directors
        directors = self.actor_params["directors"]
        self.directors = dict()
        self.directors_allow = dict()

        for dr_name, dr_params in directors.items():
            
            self.default_models = {role: params["classes"][0] for role, params in dr_params['roles'].items()}
            self.default_interfaces = {role: self.model_to_interface[model] for role, model in self.default_models.items()}

            tmp_params = copy.copy(self.parameters)

            for role, params in dr_params["roles"].items():
                tmp_params["COACH_params"]["Agents"][role] = self.model_params[params["classes"][0]]

            dr_class = directors_module.__dict__.get(dr_params["class_name"])
            self.directors[dr_name] = dr_class(
                env_creator = self.env_creator,
                COACHEnvClass = self.COACHEnvClass, 
                params = tmp_params, 
                model_path = dr_params["path"]
            )

            class_iter = product(*[classes["classes"] for classes in dr_params["roles"].values()])

            for t in class_iter:
                self.directors_allow[tuple(t)] = self.directors[dr_name]
                self.model_to_interface
            
    def generator_available(self):
        return tuple(self.plans.current.models.values()) in self.directors_allow.keys()
        

    def generate_plan(self):
        current_models = self.plans.current.models
        director = self.directors_allow[tuple(current_models.values())]
        coas, traj = director.generate_coas(self.parameters)

        tmp = self.plans.new_plan()
        tmp.coas = coas
        tmp.trajectory = traj

        tmp.interfaces = copy.copy(self.plans.current.interfaces)
        tmp.models = copy.copy(self.plans.current.models)
        tmp.model_classes = copy.copy(self.plans.current.model_classes)
        tmp.model_params = copy.copy(self.plans.current.model_params)


    def run_plan(self, plan):
        logger.debug("env_factory: run_plan")
        logger.debug("\t plan info: %s %s %s", plan.name, plan.interfaces, plan.models)

        if self.render_gif:
            traj = self.coach_env.get_traj_from_plan(plan, render=True)
            frames = np.array(traj.frames)
            plan.visualizations = px.imshow(frames, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
        else:
            traj = self.coach_env.get_traj_from_plan(plan, render=False)

            plan.visualizations = self.coach_env.plot_trajectory_component(
                traj, 
                plan=plan, 
                env=self.coach_env.env, 
                render_mode="plotly"
            )
        
        plan.trajectory = traj
        plan.telemetry = self.coach_env.trajectory_telemetry(traj, plan, self.coach_env.env)
        
    
class Plan:
    id = 0

    @staticmethod
    def fromCOAs(coas, name=None):
        plan = Plan(coas.keys())
        if name is not None:
            plan.name = name

        plan.coas = coas
        return plan

    def __init__(self, 
        roles, 
        name = None, 
        interfaces_by_role=None, 
        model_classes_library=None,
        model_params_library=None
        ):
        self.id = copy.copy(Plan.id)
        if name is None:
            name = f"Plan_{Plan.id}"
            Plan.id += 1
        
        self.interfaces_by_role = interfaces_by_role
        self.model_classes_library = model_classes_library
        self.model_params_library = model_params_library

        self.name = name
        self.roles = list(roles)

        self.coas = {role: COA() for role in self.roles}
        self.interfaces = {role: None for role in self.roles}
        self.models = {role: None for role in self.roles}
        self.model_classes = {role: None for role in self.roles}
        self.model_params = {role: dict() for role in self.roles}
        self.trajectories = {role: None for role in self.roles}
        
        self.visualizations = None
        self.telemetry = None

    def __getitem__(self, role):
        return self.coas[role]

    def items(self):
        return self.coas.items()
    
    def values(self):
        return self.coas.values()
    
    def keys(self):
        return self.coas.keys()

    def set_interface(self, role, interface):
        if hasattr(interface, "__len__"):
            if len(interface) == 0:
                interface = None
        self.interfaces[role] = interface
        self.models[role] = None
        self.coas[role] = COA()

    def get_interface(self, role):
        interface = self.interfaces.get(role,None)
        instance = self.interfaces_by_role[role][interface]

        return interface, instance
    
    def set_model(self, role, model):
        self.models[role] = model
        self.model_classes[role] = self.model_classes_library[self.interfaces[role]][model]
        self.model_params[role] = self.model_params_library[self.interfaces[role]][model].get("params", dict())

    def get_model(self, role):
        return self.models[role]
    
    def get_timelines(self):
        return {role: self.coas[role].to_dict() for role in self.roles}
    
    def get_dash_timelines(self):
        timelines = self.get_timelines()
        for k,v in timelines.items():
            if len(v) == 0:
                timelines[k] = []
            else:
                timelines[k] = list(v.keys())
        return timelines

    
    def new_event(self, role):
        logger.debug("Adding new event to COA: %s", self.coas[role])
        action_name, act_box = list(self.get_interface(role)[1].action_dictionary.items())[0]

        ## [{"start":int, "action": str, "parameters":np.array}]
        last_t = -1
        if len(list(self.coas[role].timeline.keys()))>0:
            last_t = list(self.coas[role].timeline.keys())[-1]

        logger.debug("Action Name: %s", action_name)

        tmp = TimelineEvent(
            label=action_name,
            parameters=copy.copy(act_box.default),
            tags=role
        )
        
        self.coas[role].add_event(time=last_t + 1, event=tmp)

        logger.debug("Added new event to COA: %s", self.coas[role])


class PlanLibrary:
    def __init__(self, 
        env_factory, 
        plans = [],
        default_interfaces=None,
        default_models=None
        ):
        self.roles = env_factory.roles
        self._plans: list[Plan] = plans
        self.current: Plan = None
        self.active = set()
        self.interfaces_by_role = env_factory.interfaces_by_role
        self.model_classes_library = env_factory.model_classes
        self.model_params_library = env_factory.model_params_by_interface

        if not default_interfaces:
            self.default_interfaces = {role: env_factory.get_interfaces_by_role(role)[0] for role in self.roles}
        else:
            self.default_interfaces = default_interfaces
        
        if not default_models:
            self.default_models = {role: env_factory.get_interface_models(self.default_interfaces[role])[0] for role in self.roles}
        else: 
            self.default_models = default_models

        self.new_plan()
        self.current = self.all()[0]
        self.active.add(self.current.id)

    def ids(self):
        return list(self._id_to_plan.keys())
    
    def get(self, id):
        return self._id_to_plan[id]
    
    def set_current_plan(self, id):
        self.current = self._id_to_plan[id]
    
    def new_plan(self, name=None):
        tmp = Plan(
            self.roles, 
            name=name, 
            interfaces_by_role = self.interfaces_by_role, 
            model_classes_library=self.model_classes_library,
            model_params_library=self.model_params_library   
        )

        for role, interface in self.default_interfaces.items():
            tmp.set_interface(role, interface)
            tmp.set_model(role, self.default_models[role])
        
        self.add_plan(tmp)
        return tmp

    def add_plan(self, plan: Plan):
        self._plans.append(plan)
        self._id_to_plan = {plan.id:plan for plan in self._plans}

    def all(self):
        return self._plans


        
class COALibrary:
    def __init__(self, roles, coas: dict[str, COA] = None):
        self.roles = roles
        self.role_types = list(set([role_type(role) for role in roles]))

        self._library = {role_type:[] for role_type in self.role_types}

        if coas is not None:
            for role_tag, coa in coas.items:
                self.add_coa(role_tag, coa)


    def add_coa(self, role_tag, coa):
        if role_tag in self.roles:
            role_type = role_tag
        elif role_type(role_tag) in self.roles:
            role_type = role_type(role_tag)
        else:
            raise Exception(f"Role type {role_tag} cannot be found. Valid role types are {list(self._library.keys())}.")
                
        if coa not in self._library[role_type]:
            self._library[role_type].append(coa)

    def __setitem__(self, key, item):
        self._library[key] = item

    def __getitem__(self, key):
        return self._library[key]

    def values(self):
        return self._library.values()

    def items(self):
        return self._library.items()
    
    def keys(self):
        return self._library.keys()
