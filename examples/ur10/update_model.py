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
