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

class callback_tools:
    env_factory = None
    
    @staticmethod
    def list_to_role(entry_list):
        for i, entry in enumerate(entry_list):
            if entry is not None:
                return i, callback_tools.env_factory.roles[i], entry

    @staticmethod
    def make_role_returns(role, r):
        returns = [None]*len(callback_tools.env_factory.roles)
        i = callback_tools.env_factory.roles.index(role)
        returns[i] = r
        return(returns)
    
    @staticmethod
    def get_role_from_callback(role, callback):
        # print("Roles:", env_factory.role_to_idx)
        return callback[callback_tools.env_factory.role_to_idx[role]]

    @staticmethod
    def make_returns_from_dict(r):
        returns = [None]*len(callback_tools.env_factory.roles)
        for i, role in enumerate(callback_tools.env_factory.roles):
            if role in r.keys():
                returns[i] = r[role]
        
        return callback_tools.nones_to_empty_list(returns)

    @staticmethod
    def make_returns(r):
        return [r]*len(callback_tools.env_factory.roles)

    @staticmethod
    def nones_to_empty_list(r):
        if type(r) is list:
            return [callback_tools._ntol(v) for v in r]
        if type(r) is dict:
            return {k:callback_tools._ntol(v) for k, v in r.items()}

    @staticmethod
    def _ntol(r):
        if r is None:
            return []
        
        return r