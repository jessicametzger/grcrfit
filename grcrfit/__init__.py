import os

# for use on cluster
if os.path.exists('/users/jmetzger/local/'):
    import sys
    sys.path = ['/users/jmetzger/local/'] + sys.path

import matplotlib
matplotlib.use('agg')

# the main functions users will use
from .main import run_fit
from .analysis import walker_plot, corner_plot, bestfit_plot, get_chisqu

# classes the user might want
from .run import Run, Fitter
from .model import Model