from importlib import metadata

package = __package__ or "lightning-cv"

_data = metadata.metadata(package)
__title__: str = _data["name"]
__description__: str = _data["summary"]
__version__: str = _data["version"]
__author__: str = _data["author"]
__license__: str = _data["license"]
