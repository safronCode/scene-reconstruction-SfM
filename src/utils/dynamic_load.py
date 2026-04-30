import inspect
from src.features.base import BaseModel


def dynamic_load_model(module, model_name):
    module_path = f"{module.__name__}.{model_name}"
    module = __import__(module_path, fromlist=["*"])

    classes = inspect.getmembers(module, inspect.isclass)
    classes = [c for c in classes if c[1].__module__ == module_path]
    classes = [c for c in classes if issubclass(c[1], BaseModel)]

    assert len(classes) == 1, classes
    return classes[0][1]
