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


from __future__ import annotations

import sys
from pathlib import Path

_script_dir = str(Path(__file__).parent.resolve())
_project_root = str(Path(__file__).parents[2])
for p in [_script_dir, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

if "--update-model" not in sys.argv:
    sys.argv.insert(1, "--update-model")

from calibration import main

if __name__ == "__main__":
    main()
