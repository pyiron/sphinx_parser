import keyword
import numpy as np


indent = 4 * " "
predefined = ["description", "default", "data_type", "required", "alias", "unit"]


def find_alias(all_data, head=None):
    if head is None:
        head = []
    results = {}
    for key, data in all_data.items():
        if key == "alias":
            results["/".join(head)] = data.replace(".", "/")
        if isinstance(data, dict):
            results.update(find_alias(data, head + [key]))
    return results


def get(obj, path, sep="/"):
    for t in path.split(sep):
        obj = obj[t]
    return obj


def set(obj, path, value):
    *path, last = path.split("/")
    for bit in path:
        obj = obj.setdefault(bit, {})
    obj[last] = value


def get_safe_parameter_name(name):
    if keyword.iskeyword(name):
        name = name + "_"
    return name


def get_docstring_line(data, key):
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
            txt += 2 * indent + get_docstring_line(data, key) + "\n"
    txt += indent + '"""\n'
    return txt


def get_input_arg(key, entry, indent=indent):
    t = entry.get("data_type", "dict")
    if not entry.get("required", False):
        t = f"Optional[{t}] = None"
    t = f"{indent}{key}: {t},"
    return t


def get_function(data, tag, predefined=predefined):
    d = {get_safe_parameter_name(key): value for key, value in data.items()}
    args = "\n".join([get_input_arg(key, value) for key, value in d.items() if key not in predefined])
    docstring = get_docstring(d, d.get("description", None))
    output = "\n".join([2 * indent + f"{key}={key}," for key in d.keys() if key not in predefined])
    return f"def get_{tag}(\n{args}\n):\n" + docstring + indent + f"return fill_values(\n{output}\n{indent})"


def get_all_function_names(all_data, head=""):
    key_lst = []
    for tag, data in all_data.items():
        if tag not in predefined and data.get("data_type", "dict") == "dict":
            key_lst.append(head + tag)
            key_lst.extend(get_all_function_names(data, head=head + tag + "/"))
    return key_lst


def get_tags(tags, max_steps=10):
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
