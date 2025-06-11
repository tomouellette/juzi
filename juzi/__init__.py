from . import cs
import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['cs']})
