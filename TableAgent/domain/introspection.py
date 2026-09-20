"""Make the objects handed to a code-writing agent describe their own surface.

An agent writing code against these types guesses attribute names, and a guess that
misses costs a whole execution round to discover. The alternative to chasing each miss
with a new alias -- `.name` today, `.sheet_name` tomorrow -- is to answer the guess:
name what the object does have, and point at the nearest match. That covers the names
nobody has seen fail yet, and it keeps the type's real surface small.
"""

from __future__ import annotations

import difflib
from typing import Any


def public_attribute_names(instance: Any) -> list[str]:
    """Every attribute of `instance` an agent is meant to reach for.

    A type may declare `__agent_attributes__` to name its own surface. That is for types
    whose inherited members drown the relevant ones -- a `str` subclass carries forty-odd
    string methods, and listing them buries the three names that matter and makes the
    nearest-match suggestion pick from noise. It declares intent once; it is not a place
    to accumulate aliases for attributes a model was seen to guess.
    """
    declared = getattr(type(instance), "__agent_attributes__", None)
    if declared:
        return sorted(str(name) for name in declared)

    names: set[str] = set()
    for source in (type(instance).__mro__, ):
        for klass in source:
            if klass is object:
                continue
            names.update(
                name for name in vars(klass) if not name.startswith("_")
            )
    names.update(
        name for name in getattr(instance, "__dict__", {}) if not name.startswith("_")
    )
    for slot in getattr(type(instance), "__slots__", ()) or ():
        if isinstance(slot, str) and not slot.startswith("_"):
            names.add(slot)
    return sorted(names)


def describe_attribute_miss(instance: Any, name: str) -> str:
    """The message an unknown attribute should carry."""
    available = public_attribute_names(instance)
    close = difflib.get_close_matches(name, available, n=3, cutoff=0.4)
    message = f"{type(instance).__name__!r} object has no attribute {name!r}."
    if close:
        message += f" Closest available: {', '.join(repr(item) for item in close)}."
    message += f" Available: {', '.join(available) if available else '(none)'}."
    return message


class SelfDescribing:
    """Mixin turning an unknown attribute into the list of the real ones.

    Names starting with an underscore fall through untouched so that `copy`, `pickle`,
    pandas and IPython can probe for their private protocol hooks without collecting a
    paragraph in reply.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(describe_attribute_miss(self, name))


__all__ = ["SelfDescribing", "describe_attribute_miss", "public_attribute_names"]
