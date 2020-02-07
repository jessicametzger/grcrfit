import os

# for use on my cluster account
if os.path.exists(os.path.expanduser('~/local')):
    import sys
    sys.path = [os.path.expanduser('~/local/')] + sys.path

import matplotlib
matplotlib.use('agg')

# the main function users will use
from .main import run_fit

# classes the user might want
from .run import Run, Fitter
from .model import Model
