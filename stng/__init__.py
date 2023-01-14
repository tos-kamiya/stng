import importlib.metadata

__version__ = importlib.metadata.version("stng")

from . import scanners
from . import iter_funcs
from . import text_funcs
from . import search_result
from . import stng
