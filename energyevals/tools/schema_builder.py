import inspect
import re
import types  # types.UnionType covers the X | Y union syntax (Python 3.10+)
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

_TYPE_MAP: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
}


def get_method_description(method: Callable[..., Any]) -> str:
    """Extract the summary paragraph from *method*'s docstring.

    Everything before the first ``Args:``, ``Returns:``, ``Parameters:``,
    ``Raises:``, or ``**`` heading is treated as the summary.
    """
    doc = method.__doc__
    if not doc:
        raise ValueError(
            f"Method {getattr(method, '__name__', method)} has no docstring; "
            "a docstring is required for @tool_method methods."
        )
    lines = doc.strip().splitlines()
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(
            ("args:", "returns:", "parameters:", "raises:", "**")
        ):
            break
        if not stripped and summary_lines:
            break
        if stripped:
            summary_lines.append(stripped)
    return " ".join(summary_lines)


def _parse_docstring_args(method: Callable[..., Any]) -> dict[str, str]:
    """Extract per-parameter descriptions from the ``Args:``/``Parameters:`` section."""
    doc = getattr(method, "__doc__", None) or ""
    lines = doc.strip().splitlines()

    in_args = False
    args_dict: dict[str, str] = {}
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith(("args:", "parameters:")):
            in_args = True
            continue

        if in_args and stripped.lower().startswith(("returns:", "raises:", "**")):
            if current_param:
                args_dict[current_param] = " ".join(current_desc).strip()
            break

        if not in_args:
            continue

        if not stripped:
            if current_param:
                args_dict[current_param] = " ".join(current_desc).strip()
                current_param = None
                current_desc = []
            continue

        param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
        if param_match:
            if current_param:
                args_dict[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            current_desc = [param_match.group(2)] if param_match.group(2) else []
        elif current_param:
            current_desc.append(stripped)

    if current_param:
        args_dict[current_param] = " ".join(current_desc).strip()

    return args_dict


def python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        if all(isinstance(v, bool) for v in values):
            return {"type": "boolean", "enum": values}
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return {"type": "integer", "enum": values}
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, (int, float)) for v in values):
            return {"type": "number", "enum": values}
        return {"enum": values}

    if origin is Union or isinstance(annotation, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return python_type_to_json_schema(non_none[0])
        return {"oneOf": [python_type_to_json_schema(a) for a in non_none]}

    if origin is list:
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    return dict(_TYPE_MAP.get(annotation, {"type": "string"}))


def build_parameters_schema(method: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema ``parameters`` object from *method*'s signature and docstring.

    Combines ``inspect.signature`` for parameter names and defaults,
    ``get_type_hints`` for JSON Schema types, and the docstring ``Args:``
    section for per-parameter descriptions.
    """
    func = getattr(method, "__func__", method)

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    hints.pop("self", None)
    hints.pop("return", None)

    sig = inspect.signature(method)
    doc_args = _parse_docstring_args(method)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = hints.get(param_name, param.annotation)
        has_default = param.default is not inspect.Parameter.empty

        if annotation is not inspect.Parameter.empty:
            prop_schema = python_type_to_json_schema(annotation)
        else:
            prop_schema = {"type": "string"}

        if param_name in doc_args:
            prop_schema["description"] = doc_args[param_name]

        if has_default and param.default is not None:
            prop_schema["default"] = param.default

        properties[param_name] = prop_schema

        if not has_default:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
