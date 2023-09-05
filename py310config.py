print('setting up ipython for notebook')

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
from pathlib import Path
from tqdm.auto import tqdm

# setup hvplot and plotly
try:
    import hvplot.pandas
    import holoviews as hv
    hvkw = dict(height=500, width=700, grid=True, legend='top')
except ImportError:
    pass

try:
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "notebook"
    pio.templates.default = 'plotly_dark'
    pd.options.plotting.backend = 'plotly'
except ImportError:
    pass

try:
    from ipydatagrid import DataGrid
except ImportError:
    pass

vs_code_dark_theme_rc = {
    'axes.facecolor': '#181818',
    'figure.facecolor': '#1f1f1f',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.labelcolor': 'white',
    'legend.facecolor': '#1f1f1f',
    'legend.edgecolor': '#1f1f1f',
    'legend.framealpha': 0.9,
    'grid.color': '#252525',
    'axes.edgecolor': '#252525',
    'figure.figsize': (9, 5),
}

light_theme_rc = {
    'figure.figsize': (9, 5),
}

sns.set(rc=light_theme_rc)

print('done configuring ipython')

#  print setup info
import os
print('\nworking directory : ' + os.getcwd() + '\n')
print(ipython.banner)
