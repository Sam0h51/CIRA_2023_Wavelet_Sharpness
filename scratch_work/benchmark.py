import argparse
import matplotlib.pyplot as plt

from dataloader import generate_synthetic_data, load_data, synthetic_f
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f


parser = argparse.ArgumentParser(description='Sharpness Benchmarks')

# Data configuration
parser.add_argument('-s', '--synthetic', type=str, default=None,
                    help='generate synthetic data',
                    choices=list(synthetic_f.keys()))
parser.add_argument('-i', '--input', type=str, default='../data/kh_ABI_C13.nc',
                    help='name of input file to load data from')

# Computation configuration
parser.add_argument('-t', '--transformation', type=str, default='vflip',
                    help='transformation to perform on data',
                    choices=list(transform_d.keys()))
parser.add_argument('-m', '--metric', type=str, default='all',
                    help='evaluation metric to compute',
                    choices=['all'] + list(metric_f.keys()))

# Visualization configuration
parser.add_argument('--visualize', action='store_true',  # default false
                    help='visualize and save the operations')
parser.add_argument('-o', '--output', type=str, default='../media/output.png',
                    help='name of output file visualization')


def visualize(data, args):
    cmap = 'gray'
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(data['X'], cmap=cmap)
    axs[1].imshow(data['T'], cmap=cmap)

    axs[0].set_title('X', weight='bold')
    axs[1].set_title('T', weight='bold')
    axs[1].set_title(f'({args.transformation})', loc='left')

    for ax in axs:
        ax.axis('off')

    fig.savefig(args.output, dpi=300, bbox_inches='tight')


def main(args):
    if args.synthetic:
        X = generate_synthetic_data(args.synthetic)
    else:
        X = load_data(args.input)

    T = apply_transform(X, args.transformation)
    # print(f'{X.shape=}, {X.min()=}, {X.max()=}')
    # print(f'{T.shape=}, {T.min()=}, {T.max()=}')

    metric_name = args.metric
    if metric_name == 'all':
        metrics = compute_all_metrics(X, T)
        for metric_name, result in metrics.items():
            print(f'=> {metric_name}: {result}')
    else:
        metrics = compute_metric(X, T, metric_name)
        print(f'=> {metric_name}: {metrics}')

    data = dict()
    data['X'] = X
    data['T'] = T
    data['metrics'] = metrics

    if args.visualize:
        visualize(data, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
