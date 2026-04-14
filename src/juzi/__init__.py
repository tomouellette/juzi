# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

from . import gp
from . import pl
from . import mg
from . import ut
import sys

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["gp", "pl", "mg", "ut"]})
