import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import StringIO
import random

def generateHeatmap(file):
    df = pd.read_csv(file)
    plt.figure(figsize = (20,20))
    ax = sns.heatmap(df.corr(), annot=True, linewidths=.5) #notation: "annot" and NOT "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    img = ax.get_figure()
    rand_number = random.randint(10, 100000)
    img.savefig(f'static/images/output-{rand_number}.png')
    return f'/static/images/output-{rand_number}.png'