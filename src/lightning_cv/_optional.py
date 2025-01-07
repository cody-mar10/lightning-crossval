from importlib.util import find_spec

HAS_PYDANTIC = find_spec("pydantic") is not None
HAS_OPTUNA = find_spec("optuna") is not None

if HAS_PYDANTIC:
    import pydantic

    PYDANTIC_V2 = int(pydantic.version.VERSION.split(".")[0]) == 2

    def is_pydantic_basemodel(obj) -> bool:
        """Checks if the given input is an instance of a Pydantic BaseModel OR a subclass of it."""
        if isinstance(obj, type):
            return issubclass(obj, pydantic.BaseModel)
        return isinstance(obj, pydantic.BaseModel)

else:  # pragma: no cover <- there is no need to test this...
    PYDANTIC_V2 = False

    def is_pydantic_basemodel(obj) -> bool:
        return False
