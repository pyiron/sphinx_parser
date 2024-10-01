from typing import Optional
import numpy as np


def format_value(v, indent=0):
    if isinstance(v, bool):
        return f" = {v};".lower()
    elif isinstance(v, dict) or isinstance(v, list):
        if len(v) == 0:
            return " {}"
        else:
            return " {\n" + to_sphinx(v, indent + 1) + indent * "\t" + "}"
    else:
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return " = {!s};".format(v)


def to_sphinx(obj, indent=0):
    line = ""
    for k, v in obj.items():
        current_line = indent * "\t" + k.split("___")[0]
        if isinstance(v, list):
            for vv in v:
                line += current_line + format_value(vv, indent) + "\n"
        else:
            line += current_line + format_value(v, indent) + "\n"
    return line


def append_item(group, key, value, n_max=int(1e8)):
    if key not in group:
        group[key] = value
        return group
    else:
        for ii in range(n_max):
            if f"{key}___{ii}" not in group:
                group[f"{key}___{ii}"] = value
                return group
    raise ValueError("Too many items in group")


def fill_values(group=None, **kwargs):
    if group is None:
        group = {}
    for k, v in kwargs.items():
        while k.endswith("_"):
            k = k[:-1]
        if v is not None and v is not False:
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                for i, vv in enumerate(v):
                    group = append_item(group, k, vv)
            else:
                group = append_item(group, k, v)
    return group