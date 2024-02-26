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
import re
import sys


# Utility Libraries
import numpy as np
import copy
import json

# Import Dash Things
from dash import Dash, html, dcc, Input, Output, callback, State, ALL, no_update, ctx, MATCH
import plotly.express as px
import plotly.graph_objects as go

# Import Custom Things
from DASH.html_objects import app_layout, plan_menu, actions_display
from DASH.dash_utilities import callback_tools as ct

# Import Env Things:
from DASH.coach_integration import COACHIntegration
from coach import COACHEnvironment

# Logging
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

pymunk_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("pymunk")]

for log_handler in pymunk_loggers:
    log_handler.setLevel(logging.INFO)

##########################################################
# Inputs
##########################################################


### Plan Selector
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type":"plan", "id": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

def select_plan(plan):   
    logger.debug("INPUT: select_plan") 
    
    plan_id = ctx.triggered_id['id']
    logger.debug("\t %s %s %s","select_plan", plan, plan_id)
    if plan[plan_id] > 0:
        ct.env_factory.plans.set_current_plan(plan_id)

    return True



### Plan View Checkboxes
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type":"plan_view", "id": ALL}, "value"),
    prevent_initial_call=True,
)
def view_plan(plan_checks):
    logger.debug("INPUT: view_plan") 
    
    plan_id = ctx.triggered_id['id']
    logger.debug("\t %s %s %s", "view_plan", plan_checks, plan_id)
    checked = len(plan_checks[plan_id])>0

    if checked:
        ct.env_factory.plans.active.add(plan_id)
        logger.debug("\t %s %s", "plan added, active plans:", ct.env_factory.plans.active)
    else:
        if plan_id in env_factory.plans.active:
            env_factory.plans.active.remove(plan_id)
            logger.debug("\t %s %s", "plan removed, active plans:", ct.env_factory.plans.active)
        else:
            return no_update

    return True

### Interface Selector
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type": "agent_interface", "role": ALL}, "value"),
    prevent_initial_call=True,
)
def choose_interface(agent_interface):   
    logger.debug("INPUT: choose_interface") 

    # Get the role that changed. 
    role = ctx.triggered_id['role']
    interface = ct.get_role_from_callback(role, agent_interface)
    logger.debug("\t %s %s","choose_interface:", role)

    logger.debug("\t %s %s %s %s %s %s","Interface Selector:", role, "new interface:", interface, "old interface:", ct.env_factory.plans.current.get_interface(role)[0])
    if interface == ct.env_factory.plans.current.get_interface(role)[0]:
        logger.debug("INPUT: choose_interface - no_update") 
        return no_update
    else:
        logger.debug("INPUT: choose_interface - updated") 
        ct.env_factory.plans.current.set_interface(role, interface)
        return True



### Model Selector
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type": "onboard_model", "role": ALL}, "value"),
    prevent_initial_call=True,
)
def select_model(onboard_model):
    logger.debug("INPUT: select_model")
    role = ctx.triggered_id['role']

    model = ct.get_role_from_callback(role, onboard_model)
    if model == ct.env_factory.plans.current.get_model(role):
        return no_update
    else:
        logger.debug("\t Setting Model for Plan %s %s %s %s %s", ct.env_factory.plans.current.id, "for role", role, "to", model)
        ct.env_factory.plans.current.set_model(role, model)
        return True



### Action Parameters
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type": "actionparam", "plan": ALL, "time": ALL, "type":ALL, "index": ALL, "role": ALL}, "value"),
    State({"type": "actionparam", "plan": ALL, "time": ALL, "type":ALL, "index": ALL, "role": ALL}, "id"),
    prevent_initial_call=True,
)
def change_action_params(params, id):
    logger.debug("INPUT: change_action_params")
    logger.debug("\t %s %s", "calling id:", id)

    if ctx.triggered_id is None:
        ## This got called by a plan change
        return no_update

    logger.debug("\t %s %s", "ctx.triggered_id:", ctx.triggered_id)

    role = ctx.triggered_id['role']
    logger.debug("\t %s %s %s %s","New Params:", params, "role:", role)
    
    locks["action_card"] = True
    logger.debug("\t %s","LOCKING ACTION PARAMS")
    
    i = id.index(ctx.triggered_id)
    plan = ct.env_factory.plans.get(ctx.triggered_id["plan"])
    coa = plan.coas[ctx.triggered_id["role"]]
    event = coa.get(time = ctx.triggered_id["time"], label = ctx.triggered_id["type"])

    logger.debug("\t %s %s","Param_Update: Current COA" , coa)

    event.parameters[ctx.triggered_id["index"]] = params[i]
    
    logger.debug("\t %s %s","Param_Update: New COA" , coa)

    return True

### Timeline Selector

@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type": "timeline", "role": ALL}, "value"),
    prevent_initial_call=True,
)

def edit_timeline(new_timelines):
    role = ctx.triggered_id['role']

    logger.debug("\t %s","INPUT: edit_timeline")
    logger.debug("\t %s %s","new_timelines", new_timelines)
    logger.debug("\t %s %s","Old COA:", ct.env_factory.plans.current.coas[role])

    # Figure out which timeline event changed    
    changed_timeline = ct.get_role_from_callback(role, new_timelines)
    logger.debug("\t %s %s %s","updated timeline:", role, changed_timeline)

    if len(changed_timeline)==0:
        logger.debug("INPUT: edit_timeline - no_update")
        return no_update

    old_timeline = ct.env_factory.plans.current.get_dash_timelines()[role]

    # If we drag a timepoint across another it will change it's position in the ordering
    # so we need some extra logic around that
    old = []
    new = []

    for j_old, j_new in zip(old_timeline, changed_timeline):
        if not j_old == j_new:
            old.append(j_old)
            new.append(j_new)

    if len(old)==0:
        logger.debug("INPUT: edit_timeline - no_update")
        return no_update

    # Check if we've moved one point past another
    if len(old)>1:
        if new[0] == old[1]:
            j_new = new[1]
            j_old = old[0]
        else:
            j_new = new[0]
            j_old = old[1]
    else:
        j_new = new[0]
        j_old = old[0]

        # Check if we're moving one point on top of another
        if j_new in old_timeline:
            # Kind of dumb trick to make it bounce to the side you're dragging it from. 
            sign = -(j_new - j_old)/abs(j_new - j_old) 
            while j_new in changed_timeline:
                j_new += sign*1
            
    logger.debug("\t %s %s","New Timelines:", changed_timeline)
    logger.debug("\t %s %s","Old Timelines:", old_timeline)
    logger.debug("\t %s %s","Change Index:", j_old, j_new)

    ct.env_factory.plans.current.coas[role].move_events(time=j_old, to=j_new)

    logger.debug("\t %s %s","New COA:", ct.env_factory.plans.current.coas[role])

    logger.debug("INPUT: edit_timeline - Update Backend")
    return True

##########################################################
# Buttons
##########################################################

### New Plan
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input("button_new_plan", "n_clicks"),
    prevent_initial_call=True,
)
def new_plan(nclicks):
    logger.debug("BUTTON: new_plan")
    ct.env_factory.plans.new_plan()

    return True

### New Command
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input({"type":"button_add_new_command", "role":ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def new_command(nclicks):
    logger.debug("BUTTON: new_command")
    role = ctx.triggered_id['role']
    current_plan = ct.env_factory.plans.current
    current_plan.new_event(role)

    locks["action_card"] = True

    return True

### Simulate Plan
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input("button_simulate_plan", "n_clicks"),
    prevent_initial_call=True,
)
def simulate_plan(nclicks):
    logger.debug("BUTTON: simulate_plan")
    current_plan = ct.env_factory.plans.current
    ct.env_factory.run_plan(current_plan)

    return True



### Generate Plan
@callback(
    Output("update_backend_coa", "data", allow_duplicate=True),
    Input("button_generate_plan", "n_clicks"),
    prevent_initial_call=True,
)
def simulate_plan(nclicks):
    logger.debug("BUTTON: generate_plan")
    ct.env_factory.generate_plan()
    return True


##########################################################
# Outputs
##########################################################

@callback(
    Output({"type": "onboard_model", "role": ALL}, "options", allow_duplicate=True),
    Output({"type": "onboard_model", "role": ALL}, "value", allow_duplicate=True),
    Output({"type": "timeline", "role": ALL}, "value", allow_duplicate=True),
    Output({"type": "agent_interface", "role": ALL}, "value", allow_duplicate=True),
    Output({"type": "actions_card", "role": ALL}, "children", allow_duplicate=True),
    Output("table_plans", "children", allow_duplicate=True),
    Output('button_generate_plan', 'disabled'),
    Input("update_frontend_elements", "data"),
    prevent_initial_call=True,
)

def display_choose_model(update):
    logger.debug("OUTPUT: display_choose_model")

    possible_models = ct.make_returns_from_dict(ct.env_factory.get_current_models())
    timelines = ct.make_returns_from_dict(ct.env_factory.plans.current.get_dash_timelines())
    interface = ct.make_returns_from_dict(ct.env_factory.plans.current.interfaces)
    actions = actions_display(ct.env_factory)
    plans = plan_menu(ct.env_factory)
    generator = not ct.env_factory.generator_available()

    selected_model = [ct.env_factory.plans.current.models[role] for role in ct.env_factory.roles]

    logger.debug("\t %s %s","Models", possible_models)
    logger.debug("\t %s %s","Selected Model", selected_model)
    logger.debug("\t %s %s","Timelines", timelines)
    logger.debug("\t %s %s","Interfaces", interface)

    # return possible_models, Tb_children, timelines, interface
    return possible_models, selected_model, timelines, interface, actions, plans, generator


## Update COA Visualizations
@callback(
    Output("plotly_visialization", "figure", allow_duplicate=True),
    Input('update_frontend_elements', 'data'),
    prevent_initial_call=True,
)

def visualization(update_frontend_elements):
    logger.debug("OUTPUT: visualization")
    if ct.env_factory.plans.current.visualizations is not None:
        return ct.env_factory.plans.current.visualizations
    else:
        return no_update
    


## Update Telemetry

@callback(
    Output("stats-graphic-1", "figure", allow_duplicate=True),
    Output("stats-graphic-2", "figure", allow_duplicate=True),
    Output("stats-graphic-3", "figure", allow_duplicate=True),
    Input('update_frontend_elements', 'data'),
    prevent_initial_call=True,
)

def telemetry(update_frontend_elements):
    logger.debug("OUTPUT: telemetry")
    figures = []

    for plan_id in ct.env_factory.plans.active:
        logger.debug("\t %s %s", "active plan_id", plan_id)
        plan = ct.env_factory.plans.get(plan_id)

        if plan.telemetry is not None:
            for i, t in enumerate(plan.telemetry[:3]):
                if len(figures) <= i:
                    logger.debug("\t %s %s %s", "adding figure", len(figures), i)
                    f = go.Figure()
                    f.update_layout(
                        margin={"l": 5, "b": 0, "t": 0, "r": 80},
                        hovermode="closest",
                        showlegend=False,
                        xaxis_title=t.xlabel, 
                        yaxis_title=t.ylabel,
                    )
                    figures.append(f)

                t = t.as_df()
                opacity = .2
                if plan_id == ct.env_factory.plans.current.id:
                    logger.debug("\t %s %s %s", "active plan_id", plan_id, "this plan is the current one.")
                    opacity = 1

                for col in t.columns:
                    figures[i].add_trace(go.Scatter(x=t.index,y=t[col],
                        mode='lines',
                        name=f"{plan_id}_{col}",
                        opacity=opacity
                        ))
    
    if len(figures) == 0:
        return no_update
    
    while len(figures)<3:
        figures.append(go.Figure())

    return figures
        
##########################################################
# Update Backend
##########################################################

@callback( 
    Output("update_frontend_elements", "data", allow_duplicate=True),
    Input("update_backend_coa", "data"),
    # prevent_initial_call=True,
    prevent_initial_call='initial_duplicate'
)

def update_coa(coa_update):
    logger.debug("BACKEND: update_coa")
    return True

##########################################################
# Setup
##########################################################

locks = {
            "action_card": False
        }

if __name__ == "__main__":
    logger.debug("app - ################### RELOADING DASH APP #########################")
    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    env_factory = env_creator(get_env_class)
    ct.env_factory = env_factory

    ## App Layout
    app = Dash(__name__)
    app.layout = app_layout(env_factory)
    app.run(debug=True)