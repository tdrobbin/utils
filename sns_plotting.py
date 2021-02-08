import matplotlib.pyplot as plt
import seaborn as sns


def plotting(grid=True, dark=False):
    plt.rcdefaults()
    style = 'darkgrid'
    
    if (grid) and (dark):
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
    
    elif (grid) and (not dark):
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
        
    elif (not grid) and (dark):
        rc={
            'figure.figsize': (9, 6), 
            'axes.spines.top': False, 
            'axes.spines.right': False,
            'axes.facecolor': '#111111',
            'figure.facecolor': '#111111',
            "lines.linewidth": 1.5,
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'axes.titlepad': 18,
            'axes.edgecolor': '#dddddd',
            'legend.frameon': False,
            'axes.labelcolor': 'white'
        }
    
        style = 'ticks'
    
    elif (not grid) and (not dark):
        rc={
            'figure.figsize': (9, 6), 
            'axes.spines.top': False, 
            'axes.spines.right': False,
            'legend.frameon': False,
        }
        style = 'ticks'
    
    sns.set_theme(
        context='notebook',
        style=style,
        font_scale=1.25,
        color_codes=True,
        rc=rc
    )
