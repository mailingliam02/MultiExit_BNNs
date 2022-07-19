import os
from pathlib import Path

PROJ_DIR = Path(os.path.realpath(__file__)).parent
RUNS_DB_DIR = PROJ_DIR / 'runs_db'

def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic