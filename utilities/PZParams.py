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


"""PettingZoo Parameter Class """

import argparse
import logging
from argparse import Namespace  # for type hinting
from collections.abc import Callable  # for type hinting
from typing import Any, Optional  # for type hinting

class EnvParams:
    """Base class for environment params."""

    def __init__(self, args: Any, param_section_name: str = "env_params") -> None:
        self.args = args
        self._params = getattr(args, param_section_name, {})

    def __getitem__(self, key: str) -> Any:
        """Return value stored for key from the params dict."""
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value for key in the params dict."""
        self._params[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return value for key from the params dict, or None if it doesn't exist."""
        try:
            return self._params[key]
        except KeyError:
            return default

    def get_mutated_params(self) -> "EnvParams":
        """Return a mutated copy of the params"""
        raise NotImplementedError(
            f"get_mutated_params has not been implemented in {type(self)}"
        )

    def checkpoint(self, folder: str) -> None:
        """Save a checkpoint in the given folder."""
        raise NotImplementedError(
            f"checkpoint has not been implemented in {type(self)}"
        )

    def reload(self, folder: str) -> None:
        """Read a checkpoint from the given folder."""
        raise NotImplementedError(f"reload has not been implemented in {type(self)}")


################################################################################
## Auxiliary Functions
################################################################################
def get_env_param_class(args: Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env parameter objects
    """
    return PettingZooEnvParams


################################################################################
## Main Class
################################################################################
class PettingZooEnvParams(EnvParams):
    """Parameters for PettingZoo Environments"""
