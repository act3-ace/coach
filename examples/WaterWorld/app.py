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
import glob
import yaml
sys.path.insert(0, "../../")

from DASH.app import *

import pettingzoo.sisl.waterworld_v4 as TrainingEnv
from utilities.PZWrapper import PettingZooEnv
import examples.WaterWorld.agents as WaypointAgentModule
import examples.WaterWorld.director_env as DirectorEnv

# Logging
import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

pymunk_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("pymunk")]
for log_handler in pymunk_loggers:
    log_handler.setLevel(logging.INFO)
    
def get_env():
    return PettingZooEnv(PZGame=TrainingEnv)

if __name__ == "__main__":
    print("app - ################### RELOADING DASH APP #########################")

    with open(os.path.join("Experiment","train_director.yaml"), "r") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


    with open(os.path.join("Experiment","dash_app.yaml"), "r") as f:
        try:
            params["actor_params"] = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    print(params)

    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    env_factory = COACHIntegration(
        env_creator=get_env, 
        COACHEnvClass=COACHEnvironment,
        COACH_PettingZoo=DirectorEnv,
        parameters=params,
        agents_module=WaypointAgentModule,
        render_gif=True
        )

    ct.env_factory = env_factory

    locks = {
                "action_card": False
            }

    ## App Layout
    # print(os.environ["VSCODE_PROXY_URI"])
    # proxy_url = re.sub("{{port}}", "8050", os.environ["VSCODE_PROXY_URI"])
    # proxy_url = "/" + "/".join(proxy_url.split("/")[3:])
    # print(proxy_url)
    # proxy_url = re.sub(os.environ["ACEHUB_BASEURL"], "", proxy_url)
    app = Dash(serve_locally=True)

    app.layout = app_layout(env_factory)

    app.run(debug=True)