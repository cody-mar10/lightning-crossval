from importlib import metadata

_data = metadata.metadata(__package__)
__title__: str = _data["name"]
__description__: str = _data["summary"]
__version__: str = _data["version"]
__author__: str = _data["author"]
__license__: str = _data["license"]
