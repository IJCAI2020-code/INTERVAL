import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

sns.set_context("paper")
sns.set(
    font='serif',
)
sns.set_style("dark", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})

def vis_hist():
    data = pd.read_csv('data/layers_result.csv')
    ax = sns.barplot(
        x="layers",
        y="PR-AUC",
        hue="cohort",
        data=data,
        ci="sd",
        palette="deep",
        # hue_order=['LP', 'MLP', 'DeepWalk', 'GNN', 'GLENN', 'GENN']
        # colors=get_colors()
    )
    ax.set_ylim(0.2, 0.7)
    ax.set_yticks(np.arange(0.2, 0.7, 0.1))
    ax.set_title(
        'Performance (w.r.t PR-AUC) with Different Number of Layers')
    ax.set_xlabel('# of layers')
    ax.set_ylabel('PR-AUC Score')

    return plt.gcf(), ax


if __name__ == "__main__":
    fig, ax = vis_hist()

    sns.despine()
    sns.despine(top=True, left=True)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('./cor_hist.png')

    plt.close()