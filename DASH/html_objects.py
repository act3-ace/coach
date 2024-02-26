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

from dash import Dash, html, dcc, Input, Output, callback, State

def app_layout(env_factory):
    ## The role interface select
    RoleInterfaceSelect = []

    for role in env_factory.roles:
        RoleInterfaceSelect.append(html.Div(
                [
                    html.Div([
                        role, 
                        dcc.Dropdown(
                            env_factory.get_interfaces_by_role(role), 
                            placeholder=f"Interface",
                            id={"type": "agent_interface", "role": role}
                            ),
                        dcc.Dropdown(
                            id={"type": "onboard_model", "role": role},
                            placeholder=f"Model"
                            ),
                        html.Button("Add New Command", id={"type": "button_add_new_command", "role": role}),
                        ],
                        className="left_card",
                        style={"float": "left", "margin": "auto"},
                    ),
                    html.Div(
                        className="right_card",
                        id={"type": "actions_card", "role": role},
                        style={"float": "left", "margin": "auto"},
                    ),
                ],
                className="row agent_card",
            ))

    RoleInterfaceSelect = html.Div(RoleInterfaceSelect, className="action_cards")

    ## The Role Timeline Selection
    TimeLines = []
    for role in env_factory.roles:
        TimeLines += [
            html.Tr([
                html.Td(role, style={"float": "left", "margin": "auto", "width": 100}),
                html.Td(
                    dcc.RangeSlider(
                        0,
                        100,
                        value=[],
                        tooltip={"placement": "bottom", "always_visible": True},
                        id={"type": "timeline", "role": role},
                    ),
                style={"margin": "auto", "width": "100%"}
                ), 
            ])
        ]

    if len(TimeLines)==1:
        TimeLines = TimeLines[0]

    TimeLines = html.Table(TimeLines)

    ## App Layout
    app_layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="plotly_visialization",
                                style={"width": "600px", "height": "600px"},
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="stats-graphic-1", style={"height": "150px"}),
                            dcc.Graph(id="stats-graphic-2", style={"height": "150px"}),
                            dcc.Graph(id="stats-graphic-3", style={"height": "150px"}),
                        ],
                        style={"width": "48%", "float": "right", "display": "inline-block"},
                    ),
                ]
            ),
            TimeLines,
            html.Button("New Plan", id="button_new_plan"),
            html.Button("Generate Plan", id="button_generate_plan"),
            html.Button("Simulate Plan", id="button_simulate_plan"),
            ## These are True False values, they signle that the corresponding object should be updated.
            dcc.Store(id="update_backend_coa"),
            dcc.Store(id="update_frontend_elements"),
            ### Agent Card:
            html.Div(RoleInterfaceSelect),
            html.Div([
                "Plans",
                html.Table(id="table_plans")
            ], id="div_plans", style = {"width":"15%", "float": "right", "display": "inline-block", "padding":"10px", "border":"1px solid black"})    
        ],
    )

    return app_layout


def action_table_row(time, event, interface, plan="", role=""):
    CELL_WIDTH = "80px"

    ## Action Label:
    html_objects = []

    ## Create the header describing the actions
    action_head = [
        html.Td("", style={"padding-left": "25px", "width": "200px"}),
        html.Td("Time", style={"width": CELL_WIDTH, 'font-size': 10, 'font-style': 'italics'}),
    ]

    action_params = interface.action_dictionary[event["label"]]

    for param in action_params.items():
        # Give the name in the heading
        action_head.append(html.Td(param.description, style={"width": CELL_WIDTH, 'font-size': 10, 'font-style': 'italics'}))
    
    # Store header
    html_objects.append(html.Tr(action_head))

    TDs = [
        html.Td(
            [
                html.Div(
                    className=f"rc-slider-handle rc-slider-handle-{time}",
                    style={"margin-top": "0px", "margin-right": "5px"},
                ),
                html.Span(event["label"], style={"padding-left": "25px"}),
            ],
            style={"width": "200px"},
        )
    ]

    TDs.append(
        html.Td(
            dcc.Input(
                id={
                    "type": "actionstart", 
                    "plan": plan, 
                    "time": time, 
                    "type": event['label'], 
                    "role": role
                },
                type="number",
                value=time,
                placeholder=f"Step",
                style={"width": CELL_WIDTH},
                readOnly=True,
                debounce = True,
            )
        )
    )

    # print("Event Parameters:", event["parameters"])

    for i, param in enumerate(action_params.items()):
        for j in range(param.shape[0]):
            TDs.append(
                html.Td(
                    dcc.Input(
                        id={
                            "type": "actionparam", 
                            "plan": plan, 
                            "time": time, 
                            "type": event['label'], 
                            "index": i,
                            "role": role
                        },
                        type="number",
                        min=param.low[j],
                        max=param.high[j],
                        value=event["parameters"][i],
                        placeholder=f"[{param.low[j]:.3}, {param.high[j]:.3}]", # the :.3 is the number of significant figures
                        style={"width": CELL_WIDTH},
                        debounce = True,
                    )
                )
            )

    # print("Event Parameters:", TDs)
    html_objects.append(html.Tr(TDs))

    return html_objects


def actions_display(env_factory):
    current_plan = env_factory.plans.current
    html_objects = []
    for role in env_factory.roles:
        ## Render COA Actions
        print("\t",f"Calling Model Card for {role}")

        timelines = env_factory.plans.current.get_timelines()
        interface_name, interface = current_plan.get_interface(role)

        html_sub_objects = []

        for i, (time, events) in enumerate(timelines[role].items()):
            for j, event in enumerate(events.values()):
                html_sub_objects += action_table_row(time, event, interface, plan=current_plan.id, role=f"{role}")

        html_objects.append(html.Div(html_sub_objects))

    return html_objects


def plan_menu(env_factory):
    html_objects = []
    for plan in env_factory.plans.all():
        if plan == env_factory.plans.current:
            display_class = "current_plan"
            style = {"background-color":"#ccc", "border":0, "margin":0, "padding": "10px 40px 10px 40px"}
        else:
            display_class = "archived_plan"
            style = {"background-color":"#eee", "border":0, "margin":0, "padding": "10px 40px 10px 40px"}
        
        print("\t","plan_menu: plan", plan.name, display_class)

        active = []
        if plan.id in env_factory.plans.active:
            active = [""]

        html_objects.append( html.Tr(
                [
                    html.Td(html.A(plan.name, id={"type":"plan", "id": plan.id}, n_clicks=0), className=display_class, style=style),
                    html.Td(dcc.Checklist([""], active, id={"type":"plan_view", "id":plan.id}))
                ]
            ))
        
    return html.Tbody(html_objects) 