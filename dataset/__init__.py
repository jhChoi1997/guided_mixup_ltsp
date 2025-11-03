import importlib
import os

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        importlib.import_module(f"dataset.{filename[:-3]}")
