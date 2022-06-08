import os
from pathlib import Path

PROJ_DIR = Path(os.path.realpath(__file__)).parent
RUNS_DB_DIR = PROJ_DIR / 'runs_db'
