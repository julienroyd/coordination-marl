from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import seaborn as sns

sns.set()

def bar_chart(ax, scores, err_up=None, err_down=None, capsize=10., colors=None, group_names=None, xlabel="", ylabel="", title="", cmap="viridis"):
    # data to plot
    n_groups = len(list(scores.values())[0])

    # chooses colors
    if colors is None:
        cm = plt.cm.get_cmap(cmap)
        colors = {alg_name: np.array(cm(float(i) / float(n_groups))[:3]) for i, alg_name in enumerate(scores.keys())}

    if err_up is None:
        err_up = {alg_name: None for alg_name in scores.keys()}

    if err_down is None:
        err_down = {alg_name: None for alg_name in scores.keys()}

    # create plot
    bar_width = (1. / n_groups) * 2. * len(scores.keys())
    index = np.arange(n_groups) * (float(len(scores.keys())) + 1) * bar_width

    for i, alg_name in enumerate(scores.keys()):
        ax.bar(index + i * bar_width, scores[alg_name].values(), bar_width,
               yerr=[err_down[alg_name].values(), err_up[alg_name].values()],
               ecolor="cyan", capsize=capsize, color=colors[alg_name], label=alg_name)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')

    if group_names is not None:
        plt.xticks(index, group_names)

    ax.legend(loc='upper right')


def plot_curves(ax, ys, xs=None, colors=None, labels=None, xlabel="", ylabel="", title="", stds=None, cmap='viridis'):
    if xs is None:
        xs = [range(len(y)) for y in ys]

    if colors is None:
        cm = plt.cm.get_cmap(cmap)
        colors = [np.array(cm(float(i) / float(len(ys)))[:3]) for i in range(len(ys))]

    if labels is None:
        labels = [f'curve {i}' for i in range(len(ys))]

    # Plots losses and smoothed losses for every agent
    for i, (x, y) in enumerate(zip(xs, ys)):
        if stds is None:
            ax.plot(x, y, color=colors[i], alpha=0.3)
            ax.plot(x, smooth(y), color=colors[i], label=labels[i])
        else:
            ax.plot(x, y, color=colors[i], label=labels[i])
            ax.fill_between(x, y - stds[i], y + stds[i], color=colors[i], alpha=0.1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')


def plot_sampled_hyperparams(ax, param_samples):
    cm = plt.cm.get_cmap('viridis')
    for i, param in enumerate(param_samples.keys()):
        args = param_samples[param], np.zeros_like(param_samples[param])
        kwargs = {'linestyle': '', 'marker': 'o', 'label': param, 'alpha': 0.2,
                  'color': cm(float(i) / float(len(param_samples)))}
        if param in ['lr', 'tau', 'initial_alpha', 'grad_clip_value', 'lamda1', 'lamda2']:
            ax[i].semilogx(*args, **kwargs)
        else:
            ax[i].plot(*args, **kwargs)
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[i].get_yaxis().set_ticks([])
        ax[i].legend(loc='upper right')


def plot_vertical_densities(ax, points_groups, colors=None, xlabel="", ylabel="", title="", cmap="viridis"):
    assert type(points_groups) is OrderedDict

    if colors is None:
        cm = plt.cm.get_cmap(cmap)
        colors = {key: np.array(cm(float(i) / float(len(points_groups)))[:3]) for i, key in
                  enumerate(points_groups.keys())}

    for i, (label, points) in enumerate(points_groups.items()):
        x = [i] * len(points)
        ax.plot(x, points, linestyle='', marker='o', label=label, alpha=0.2, color=colors[label])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels([""] * len(points_groups))
    ax.set_xlim(-1, len(points_groups) + 3)

    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    ax.legend(loc='upper center', framealpha=0.25, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=1)


def smooth(data_serie, smooth_factor=0.8):
    assert smooth_factor > 0. and smooth_factor < 1.
    mean = data_serie[0]
    new_serie = []
    for value in data_serie:
        mean = smooth_factor * mean + (1 - smooth_factor) * value
        new_serie.append(mean)

    return new_serie


def create_fig(axes_shape, figsize=None):
    figsize = (8 * axes_shape[1], 5 * axes_shape[0]) if figsize is None else figsize
    fig, axes = plt.subplots(axes_shape[0], axes_shape[1], figsize=figsize)
    return fig, axes
