from pathlib import Path
import argparse
import pickle
from discrete_control.baselines.utils.config import load_config_from_json
from discrete_control.baselines.utils.plots import create_fig, plot_curves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

sns.set()
sns.set_style('whitegrid')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--plot_type", type=str, choices=["independent", "all_in_one", "stacked"])
    return parser.parse_args()

def plot_from_folder(args):
    run_dir = Path('.') / args.run_name
    to_plot = [file for file in run_dir.iterdir() if '.pkl' in file.name]
    config = load_config_from_json(str(run_dir / 'config.json'))
    curves_list = []
    d_list = []
    for f in to_plot:
        with open(str(f), "rb") as fp:
            curves_list.append(pickle.load(fp))
            d_list.append(int(f.name.split('.')[0].split('_')[-1]))
            fp.close()

    sorting = np.argsort(d_list)
    d_list = np.array(d_list)[sorting]
    curves_list = np.array(curves_list)[sorting]

    if args.plot_type == "independent":
        for d, curves in zip(d_list, curves_list):
            fig, axes = create_fig((1, 1))
            xs = [np.arange(1, config.n_episodes + 1, config.test_every)] * len(curves)
            xs = xs + [np.array([1, config.n_episodes])]
            plot_curves(axes,
                        xs=xs,
                        ys=[curves[space]['mean'] for space in curves] + [np.array([-d, -d])],
                        stds=[curves[space]['err'] for space in curves] + [np.array([0, 0])],
                        labels=list(curves.keys()) + ["optimal"],
                        xlabel="Episodes",
                        ylabel="Return",
                        title=f'L = {d}')

            fig.savefig(str(run_dir / f'learning_d_{d}.png'))
            plt.close(fig)

    elif args.plot_type == "all_in_one":
        cmap = "viridis"
        fig, axes = create_fig((1, 1))
        space_dict = {'full_space': 'A_1', 'coordinated_space': 'A_2'}
        marker_dict = {'full_space': 'X', 'coordinated_space': 'o'}
        linestyle_dict = {'full_space': '-.', 'coordinated_space': '-'}
        cm = plt.cm.get_cmap(cmap)
        colors_dict = {d: np.array(cm(float(i) / float(len(d_list)))[:3]) for i, d in
                       enumerate(d_list)}
        for d, curves in zip(d_list, curves_list):
            x = np.arange(1, config.n_episodes + 1, config.test_every)
            color = colors_dict[d]
            for key in curves.keys():
                y = curves[key]['mean']
                err = curves[key]['err']
                axes.plot(x, y, linestyle=linestyle_dict[key], alpha=1., color=color,
                          label=f'{space_dict[key]}, L={d}')
                axes.fill_between(x, y - err, y + err, color=color, alpha=0.2)
        axes.set_xlabel("Episodes")
        axes.set_ylabel("Return")
        axes.legend(loc='lower right')
        axes.set_title("Returns with coordinated vs. raw on vary")

        fig.savefig(str(run_dir / f'learning_{args.plot_type}.png'))

        plt.close(fig)

    elif args.plot_type == "stacked":
        axes_shape = (len(curves_list), 1)
        fig, axes = plt.subplots(len(curves_list), 1, figsize=(8, 4))

        cmap = "viridis"
        space_dict = {'full_space': r'$\mathcal{A}$', 'coordinated_space': "$\mathcal{A}'$"}
        cm = plt.cm.get_cmap(cmap)
        colors_dict = {d: np.array(cm(float(i) / float(len(d_list)))[:3]) for i, d in
                       enumerate(d_list)}
        linestyle_dict = {'full_space': '-.', 'coordinated_space': '-'}

        for i, (d, curves) in enumerate(zip(d_list, curves_list)):
            ax = axes[i]
            color = colors_dict[d]
            x = np.arange(1, config.n_episodes + 1, config.test_every)
            for key in curves.keys():
                y = curves[key]['mean']
                err = curves[key]['err']
                ax.plot(x, y, alpha=1., color=color, linestyle=linestyle_dict[key],
                        label=rf'{space_dict[key]}, $L={d}$')
                ax.fill_between(x, y - err, y + err, color=color, alpha=0.2)
                if not i == len(curves_list) - 1:
                    ax.set_xticklabels([])
            ax.set_ylabel("Return")
            ax.legend(loc='lower right')
            if i == len(curves_list) - 1:
                ax.set_xlabel("Episodes")

        fig.savefig(str(run_dir / f'learning_{args.plot_type}.pdf'), bbox_inches="tight")

        plt.close(fig)

if __name__ == "__main__":
    args = get_args()
    plot_from_folder(args)