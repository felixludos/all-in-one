import collections
import collections.abc

# madmom 0.16.1 still imports this alias from collections on Python 3.10+.
if not hasattr(collections, 'MutableSequence'):
  collections.MutableSequence = collections.abc.MutableSequence

from .analyze import analyze
from .visualize import visualize
from .sonify import sonify
from .typings import AnalysisResult
from .config import HARMONIX_LABELS
from .utils import load_result
