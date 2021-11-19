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
from IPython.display import display, HTML
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
whitegrid_light_theme = {
    'rc': {
        'figure.figsize': (9, 6), 
        'axes.spines.top': False, 
        'axes.spines.right': False,
        'axes.spines.bottom': False, 
        'axes.spines.left': False,
    #     'grid.color': '#dddddd',
        'legend.frameon': False,
    },
    'style' = 'whitegrid',
    'font_scale' : 1.25,
    'color_codes: True,
    'context': 'notebook'
}
ticks_light_theme = {
    'rc': {
        'figure.figsize': (9, 6), 
        'axes.spines.top': False, 
        'axes.spines.right': False,
        'legend.frameon': False,
    },
    'style' = 'ticks',
    'font_scale' : 1.25,
    'color_codes: True,
    'context': 'notebook'
}
sns.set_theme(**whitegrid_light_theme)

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

def df_info(df):
    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.sample(10))

# print setup info
import os
print('\nworking directory : ' + os.getcwd() + '\n')

print(ipython.banner)

import subprocess
print(subprocess.run(['conda', 'info'], capture_output=True, shell=True).stdout.decode("utf-8"))
