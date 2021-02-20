# configs and utilities

# pylab inline + other magic
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('pylab inline')
ipython.magic('config InlineBackend.figure_format="retina"')
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

import pandas as pd
pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:.4f}'.format
pd.plotting.register_matplotlib_converters()
from pandas import DataFrame, Series
D, S = DataFrame, Series

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.25, color_codes=True, 
        rc={
            'figure.figsize': (9, 6),
            'grid.color': '#dddddd',
            'axes.titlepad': 18,
            'legend.frameon': False,
            'axes.edgecolor': '#dddddd',
            'axes.grid': True,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True
        })
#         rc={'figure.figsize': (10, 7), 'axes.spines.top': False, 'axes.spines.right': False,})

# cells can have multiple outputs
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
InteractiveShell.ast_node_interactivity = "all"

# setup hvplot
import hvplot.pandas
import holoviews as hv
hvkw = dict(height=500, width=800, grid=True, legend='top')

def df_info(df):
    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.head(10))
