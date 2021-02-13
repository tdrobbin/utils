# need to run twice:
# https://github.com/ipython/ipython/issues/11098

# everything should word in the standard base conda environment

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas import *

import numpy as np
from numpy import *

import scipy.stats as stats
from scipy.stats import *

from importlib import reload
# from dask.distributed import Client, Future

import seaborn as sns
from seaborn import *

import matplotlib.pyplot as plt
import os
import sys

# pylab inline + other magic
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('pylab inline')
ipython.magic('config InlineBackend.figure_format="retina"')
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

# cells can have multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# df
pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:.4f}'.format
pd.plotting.register_matplotlib_converters()
D, S = DataFrame, Series
d, s = DataFrame, Series

# copied from quanstats styling:
# https://github.com/ranaroussi/quantstats/blob/71fb349ddeb500ee65af3048c1bf9d481f20c415/quantstats/_plotting/core.py

def set_plotting_style(dark=False):
    if dark:
        sns.set(
            # context='talk' ,
            # style=
            context='notebook',
            font_scale=1.25,
            color_codes=True,
            rc={
                'figure.figsize': (9, 6),
                'axes.facecolor': '#111111',
                'figure.facecolor': '#111111',
                'grid.color': '#6f6f6f',
                'grid.linewidth': 0.75,
                "lines.linewidth": 1.5,
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'axes.titlepad': 18,
                'legend.frameon': False,
                'axes.edgecolor': '#6f6f6f',
                'axes.linewidth': .75,
                'patch.force_edgecolor': True,
                'axes.labelcolor': 'white'
            }
        )

    else:
        sns.set(
            # context='talk' ,
            context='notebook',
            font_scale=1.25,
            color_codes=True,
            rc={
                'figure.figsize': (9, 6),
                'axes.facecolor': 'white',
                'figure.facecolor': 'white',
                'grid.color': '#dddddd',
                'grid.linewidth': 0.75,
                "lines.linewidth": 1.5,
                'text.color': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'axes.titlepad': 18,
                'legend.frameon': False,
                'axes.edgecolor': '#dddddd',
                'axes.linewidth': .75,
                'patch.force_edgecolor': True,
                'axes.labelcolor': 'black'
            }
        )

set_plotting_style()


def df_info(df):
    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.head(10))


# reset any bultin we may have overidden
import builtins
for var in dir(builtins):
    globals()[var] = getattr(builtins, var)
