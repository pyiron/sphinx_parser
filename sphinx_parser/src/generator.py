import builtins
import keyword
import os
from typing import Any

import yaml
from black import FileMode, format_str

indent = 4 * " "
predefined = ["description", "default", "data_type", "required", "alias", "units"]


def _find_alias(all_data: dict, head: list | None = None) -> dict[str, str]:
    """
    Find all aliases in the data structure.

    Args:
        all_data (dict): The data structure.
        head (list): The current head of the data structure.

    Returns:
        dict: A dictionary with the aliases as keys and the corresponding paths as values.
    """
    if head is None:
        head = []
    results = {}
    for key, data in all_data.items():
        if key == "alias":
            results["/".join(head)] = data.replace(".", "/")
        if isinstance(data, dict):
            results.update(_find_alias(data, head + [key]))
    return results


def _replace_alias(all_data: dict) -> dict:
    for key, value in _find_alias(all_data).items():
        _set(all_data, key, _get(all_data, value))
    return all_data


def _get(obj: dict, path: str, sep: str = "/") -> Any:
    """
    Get a value from a nested dictionary.

    Args:
        obj (dict): The dictionary.
        path (str): The path to the value.
        sep (str): The separator.

    Returns:
        Any: The value.
    """
    for t in path.split(sep):
        obj = obj[t]
    return obj


def _set(obj: dict, path: str, value):
    """
    Set a value in a nested dictionary.

    Args:
        obj (dict): The dictionary.
        path (str): The path to the value.
        value (Any): The value.
    """
    *path, last = path.split("/")
    for bit in path:
        obj = obj.setdefault(bit, {})
    obj[last] = value


def _get_safe_parameter_name(name: str) -> str:
    if keyword.iskeyword(name) or name in dir(builtins):
        name = name + "_"
    return name


def _get_docstring_line(data: dict, key: str) -> str:
    """
    Get a single line for the docstring.

    Args:
        data (dict): The data.
        key (str): The key.

    Returns:
        str: The docstring line.

    Examples:

        >>> _get_docstring_line(
        ...     {"data_type": "int", "description": "The number of iterations."},
        ...     "iterations"
        ... )
        "iterations (int): The number of iterations."
    """
    line = f"{key} ({data.get('data_type', 'dict')}):"
    if "description" in data:
        line = (line + " " + data["description"]).strip()
        if not line.endswith("."):
            line += "."
    if "default" in data:
        line += f" Default: {data['default']}."
    if "units" in data:
        line += f" Units: {data['units']}."
    if not data.get("required", False):
        line += " (Optional)"
    return line


def _get_docstring(
    all_data: dict,
    description: str | None = None,
    indent: str = indent,
    predefined: list[str] = predefined,
) -> list[str]:
    txt = [indent + '"""']
    if description is not None:
        txt.append(f"{indent}{description}\n")
    txt.append(f"{indent}Args:")
    for key, data in all_data.items():
        if key not in predefined:
            txt.append(2 * indent + _get_docstring_line(data, key))
    txt.append(
        2 * indent + "wrap_string (bool): Whether to wrap string values in apostrophes."
    )
    txt.append(indent + '"""')
    return txt


def _get_input_arg(key: str, entry: dict, indent: str = indent) -> str:
    t = entry.get("data_type", "dict")
    units = "".join(entry.get("units", "").split())
    if not entry.get("required", False) and units != "":
        t = f'u(Optional[{t}], units="{units}") = None'
    elif not entry.get("required", False):
        t = f"Optional[{t}] = None"
    elif units != "":
        t = f'u({t}, units="{units}")'
    t = f"{indent}{_get_safe_parameter_name(key)}: {t},"
    return t


def _rename_keys(data: dict) -> dict:
    d_1 = {_get_safe_parameter_name(key): value for key, value in data.items()}
    d_2 = {
        key: d
        for key, d in d_1.items()
        if not isinstance(d, dict) or d.get("required", False)
    }
    d_2.update(d_1)
    return d_2


def _get_function(
    data: dict,
    function_name: list[str],
    predefined: list[str] = predefined,
    is_kwarg: bool = False,
) -> str:
    d = _rename_keys(data)
    func = []
    if is_kwarg:
        func.append(f"{indent}wrap_string: bool = True,")
        func.append(f"{indent}**kwargs")
    else:
        func.extend(
            [
                _get_input_arg(key, value)
                for key, value in d.items()
                if key not in predefined
            ]
        )
        func.append(f"{indent}wrap_string: bool = True,")
    func.append("):")
    docstring = _get_docstring(d, d.get("description", None))
    output = [f"{indent}return fill_values("]
    if is_kwarg:
        output.append(f"{2 * indent}wrap_string=wrap_string,")
        output.append(f"{2 * indent}**kwargs")
    else:
        output.extend(
            [
                f"{2 * indent}{_get_safe_parameter_name(key)}={_get_safe_parameter_name(key)},"
                for key in d.keys()
                if key not in predefined
            ]
        )
        output.append(f"{2 * indent}wrap_string=wrap_string,")
    output.append(f"{indent})")
    result = func + docstring + output
    return "\n".join(result)


def _get_all_function_names(
    all_data: dict, head: str = "", predefined: list[str] = predefined
) -> list[str]:
    key_lst = []
    for tag, data in all_data.items():
        if tag not in predefined and data.get("data_type", "dict") == "dict":
            key_lst.append(head + tag)
            key_lst.extend(_get_all_function_names(data, head=head + tag + "/"))
    return key_lst


def _get_class(all_data: dict) -> str:
    fnames = _get_all_function_names(all_data)
    txt = ""
    for name in fnames:
        names = name.split("/")
        if len(names) > 1:
            txt += f"@_func_in_func({'.'.join(names[:-1])})\n"
        txt += "@units\n"
        if len(names) > 1:
            txt += f"def _{'__'.join(names)}(\n"
        else:
            txt += f"def {'__'.join(names)}(\n"
        txt += (
            _get_function(
                _get(all_data, name),
                ["sphinx"],
                is_kwarg=names[-1] == "main",
            )
            + "\n\n"
        )
    return txt


def _get_file_content(yml_file_name: str = "input_data.yml") -> str:
    file_location = os.path.join(os.path.dirname(__file__), yml_file_name)
    with open(file_location, "r") as f:
        file_content = f.read()
    all_data = yaml.safe_load(file_content)
    all_data = _replace_alias(all_data)
    file_content = _get_class(all_data)
    imports = [
        "from functools import wraps",
        "from typing import Optional",
        "",
        "import numpy as np",
        "from semantikon.converter import units",
        "from semantikon.metadata import u",
        "",
        "from sphinx_parser.toolkit import fill_values",
        "",
        "",
        "def _func_in_func(parentfunc):",
        "    @wraps(parentfunc)",
        "    def register(childfunc):",
        "        parentfunc.__dict__[childfunc.__name__.split('__')[-1]] = childfunc",
        "        return parentfunc",
        "    return register",
    ]
    file_content = "\n".join(imports) + "\n\n\n" + file_content
    file_content = format_str(file_content, mode=FileMode())
    return file_content


def export_class(yml_file_name: str = "input_data.yml", py_file_name: str = "input.py"):
    file_content = _get_file_content(yml_file_name)
    with open(os.path.join(os.path.dirname(__file__), "..", py_file_name), "w") as f:
        f.write(file_content)


if __name__ == "__main__":
    export_class()
