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

"""
更新模型脚本 —— 把标定后的参数写回 URDF。

本质上是 ``calibration.py --update-model`` 的快捷入口，
其他参数和 ``calibration.py`` 完全一样，用 ``--help`` 可以查看。

用法::

    python update_model.py                    # 等同于 calibration.py --update-model
    python update_model.py --output <路径>    # 指定输出 URDF 的路径
    python update_model.py --verbose          # 打印详细日志
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把脚本所在目录和项目根目录加入 sys.path，方便 import
_script_dir = str(Path(__file__).parent.resolve())
_project_root = str(Path(__file__).parents[2])
for p in [_script_dir, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 自动给命令行加上 --update-model，再交给 calibration 的 main 处理
if "--update-model" not in sys.argv:
    sys.argv.insert(1, "--update-model")

from calibration import main  # type: ignore[import]  # noqa: E402

if __name__ == "__main__":
    main()
