import keyword
import numpy as np


indent = 4 * " "
predefined = ["description", "default", "data_type", "required", "alias", "unit"]


def find_alias(all_data: dict, head: list | None = None):
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
            results.update(find_alias(data, head + [key]))
    return results


def replace_alias(all_data: dict):
    for key, value in find_alias(all_data).items():
        set(all_data, key, get(all_data, value))
    return all_data


def get(obj: dict, path: str, sep: str = "/"):
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


def set(obj: dict, path: str, value):
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


def _get_safe_parameter_name(name: str):
    if keyword.iskeyword(name):
        name = name + "_"
    return name


def _get_docstring_line(data: dict, key: str):
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
    line = f"{key} ({data.get('data_type', 'dict')}): "
    if "description" in data:
        line = (line + data["description"]).strip()
        if not line.endswith("."):
            line += "."
    if "default" in data:
        line += f" Default: {data['default']}."
    if "unit" in data:
        line += f" Unit: {data['unit']}."
    if not data.get("required", True):
        line += " (Optional)"
    return line


def get_docstring(all_data, description=None, indent=indent, predefined=predefined):
    txt = indent + '"""\n'
    if description is not None:
        txt += indent + f"{description}\n\n"
    txt += indent + "Args:\n"
    for key, data in all_data.items():
        if key not in predefined:
            txt += 2 * indent + _get_docstring_line(data, key) + "\n"
    txt += indent + '"""\n'
    return txt


def get_input_arg(key, entry, indent=indent):
    t = entry.get("data_type", "dict")
    if not entry.get("required", False):
        t = f"Optional[{t}] = None"
    t = f"{indent}{key}: {t},"
    return t


def get_function(data, tag, predefined=predefined, indent=indent, preindent=0):
    d = {_get_safe_parameter_name(key): value for key, value in data.items()}
    args = "\n".join([get_input_arg(key, value) for key, value in d.items() if key not in predefined])
    docstring = get_docstring(d, d.get("description", None))
    output = "\n".join([(2 + preindent) * indent + f"{key}={key}," for key in d.keys() if key not in predefined])
    return f"def get_{tag}(\n{args}\n):\n" + docstring + (1 + preindent) * indent + f"return fill_values(\n{output}\n{indent})"


def get_all_function_names(all_data, head="", predefined=predefined):
    key_lst = []
    for tag, data in all_data.items():
        if tag not in predefined and data.get("data_type", "dict") == "dict":
            key_lst.append(head + tag)
            key_lst.extend(get_all_function_names(data, head=head + tag + "/"))
    return key_lst


def get_unique_tags(tags, max_steps=10):
    counter = np.ones(len(tags)).astype(int)
    for _ in range(max_steps):
        reduced_tags = ["_".join(tag.split("/")[-cc:]) for cc, tag in zip(counter, tags)]
        t, c = np.unique(reduced_tags, return_counts=True)
        if c.max() == 1:
            return reduced_tags
        d = dict(zip(t, c))
        for ii, tag in enumerate(reduced_tags):
            if d[tag] > 1:
                counter[ii] += 1


def get_all_functions(all_data):
    tags = get_all_function_names(all_data)
    return [
        get_function(get(all_data, address), f_name)
        for f_name, address in zip(get_unique_tags(tags), tags)
    ]
