try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

import logging

logger = logging.getLogger(__name__)


class LossyImageError(ValueError):
    pass


class SteganographyError(ValueError):
    pass
