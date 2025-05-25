import functools
from typing import Any, Callable, TypeVar, Generic

_R = TypeVar('_R') # Return type of the decorated method

class LazyProperty(Generic[_R]):
    def __init__(self, func: Callable[..., _R]) -> None:
        self.func = func
        self.name = func.__name__
        functools.update_wrapper(self, func) # Preserve name, docstring, etc.

    def __get__(self, instance: Any, owner: Any = None) -> _R:
        if instance is None:
            # Accessing from class, return descriptor itself
            # Mypy expects the return type to be _R, but in this specific case (instance is None),
            # we are returning the descriptor instance itself.
            # This is standard practice for descriptors.
            return self # type: ignore 
        
        # Use a unique name for the cached attribute to avoid collisions
        cached_attr_name = f"_lazy_{self.name}" # e.g. _lazy_input_data
        
        if cached_attr_name not in instance.__dict__:
            # Compute the value and store it in the instance's __dict__
            value = self.func(instance)
            instance.__dict__[cached_attr_name] = value
            return value
        return instance.__dict__[cached_attr_name]
