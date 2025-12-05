# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from pathlib import Path

# Configure logging at application entry point
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(name)s - %(levelname)s - %(message)s",
)

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.tiago.utils.tiago_tools import TiagoCalibration
from figaroh.tools.robot import load_robot


# load_by_urdf = False, load robot from rospy.get_param(/robot_description)
tiago = load_robot(
    "urdf/tiago_48_schunk.urdf",
    load_by_urdf=True,
    robot_pkg="tiago_description"
)

# create a calibration object from config file
# del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
tiago_calib = TiagoCalibration(tiago, "config/tiago_unified_config.yaml", del_list=[])
tiago_calib.calib_config["known_baseframe"] = False
tiago_calib.calib_config["known_tipframe"] = False

# load data file and determine parameters to be calibrated
tiago_calib.initialize()

# solve least_squares estimation
tiago_calib.solve(plotting=True, enable_logging=False)
