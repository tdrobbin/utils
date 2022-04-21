# configs and utilities

# pylab inline + other magic
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('pylab inline')
ipython.magic('config InlineBackend.figure_format="retina"')
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

# cells can have multiple outputs
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML, display_html
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:.4f}'.format
pd.plotting.register_matplotlib_converters()
from pandas import DataFrame, Series, Timestamp
D, S, T = DataFrame, Series, Timestamp

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from pprint import pprint

grid_light_theme = {
    'rc': {
        'figure.figsize': (9, 5), 
        'grid.color': '#dddddd',
        'axes.edgecolor': '#dddddd',
        'legend.frameon': False,
        'patch.force_edgecolor': True,
    },
    'style': 'whitegrid',
    'font_scale' : 1.25,
    'color_codes': True,
    'context': 'notebook'
}
ticks_light_theme = {
    'rc': {
        'figure.figsize': (9, 5), 
        'axes.spines.top': False, 
        'axes.spines.right': False,
        'legend.frameon': False,
        'patch.force_edgecolor': True,
    },
    'style': 'ticks',
    'font_scale' : 1.25,
    'color_codes': True,
    'context': 'notebook'
}
dark_theme_updates = {
    'rc': {
        'axes.facecolor': '#111111',
        'figure.facecolor': '#111111',
        'grid.color': '#6f6f6f',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': '#6f6f6f',
        'axes.labelcolor': 'white'
    }
}
grid_dark_theme = copy(grid_light_theme).update(dark_theme_updates)
ticks_datk_theme = copy(ticks_light_theme).update(dark_theme_updates)
#

sns.set_theme(**grid_light_theme)

# setup hvplot and plotly
try:
    import hvplot.pandas
    import holoviews as hv
    hvkw = dict(height=500, width=700, grid=True, legend='top')
except ImportError:
    pass

try:
    import plotly.express as px
    px.defaults.template = "gridon"
    px.defaults.width = 700
    px.defaults.height = 500
#     pd.options.plotting.backend = 'plotly'
except ImportError:
    pass

try:
    from ipydatagrid import DataGrid
except ImportError:
    pass

def df_info(df):
    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.sample(10))

def display_dfs_inline(dfs, captions=None, margin=5):
    from functools import reduce
    from IPython.display import display_html
    
    captions = [''] * len(dfs) if captions is None else captions
    stylers = [D(df).style.set_table_attributes(f'style="display:inline; margin:{margin}px;"').set_caption(c) for df, c in zip(dfs, captions)]
    display_html(reduce(lambda x, y: x + y, (s._repr_html_() for s in stylers)), raw=True)

# reset any bultin we may have overidden
import builtins
for var in dir(builtins):
    globals()[var] = getattr(builtins, var)

# # print setup info
# import os
# print('\nworking directory : ' + os.getcwd() + '\n')

# print(ipython.banner)

import subprocess
print(subprocess.run(['conda', 'info'], capture_output=True, shell=True).stdout.decode("utf-8"))
