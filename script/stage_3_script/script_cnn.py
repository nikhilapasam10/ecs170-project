'''
Stage 3 CNN experiments on MNIST, ORL, and CIFAR.

'''

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from local_code.stage_3_code.dataset_loader import Dataset_Loader
from local_code.stage_3_code.method_cnn import (
    Method_CNN_MNIST, Method_CNN_ORL, Method_CNN_CIFAR,
)
from local_code.stage_3_code.result_saver import Result_Saver
from local_code.stage_3_code.setting_pre_split import Setting_Pre_Split
from local_code.stage_3_code.evaluation_metrics import Evaluate_Metrics


SEED = 2
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'stage_3_data') + os.sep
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_3_result') + os.sep

METHOD_CLASSES = {
    'MNIST': Method_CNN_MNIST,
    'ORL': Method_CNN_ORL,
    'CIFAR': Method_CNN_CIFAR,
}

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_scores(scores):
    return ', '.join(f'{k}={v:.4f}' for k, v in scores.items()
                     if isinstance(v, (int, float, np.integer, np.floating)))


def save_learning_curve(loss_hist, acc_hist, out_file, title):
    epochs = np.arange(1, len(loss_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, loss_hist, 'b-o', markersize=3)
    ax1.set(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(acc_hist) * 100.0, 'g-o', markersize=3)
    ax2.set(title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_experiment(name, dataset_file, *,
                   single_channel,
                   model_kwargs,
                   method_cls,
                   normalize=False,
                   title=None):
    set_seed()
    title = title or f'{name} CNN Learning Curves'
 
    data_obj = Dataset_Loader(name, '',
                              use_single_channel=single_channel,
                              normalize=normalize)
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = dataset_file
 
    method_obj = method_cls(f'cnn-{name.lower()}', '', **model_kwargs)
 
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR
    result_obj.result_destination_file_name = f'{name.lower()}_prediction_result.pkl'
 
    setting_obj = Setting_Pre_Split('pre-split', '')
    eval_obj = Evaluate_Metrics('metrics', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, eval_obj)
 
    print(f'\n===== {name} =====')
    scores = setting_obj.load_run_save_evaluate()
    save_learning_curve(
        result_obj.data['train_loss_history'],
        result_obj.data['train_acc_history'],
        os.path.join(RESULT_DIR, f'{name.lower()}_learning_curve.png'),
        title,
    )
    print(f'{name} scores: {format_scores(scores)}')
    return scores


# Main per-dataset configurations
RUNS = [
    dict(name='MNIST', dataset_file='MNIST', single_channel=True,
         method_cls=Method_CNN_MNIST, model_kwargs={}),
    dict(name='ORL', dataset_file='ORL', single_channel=True,
         method_cls=Method_CNN_ORL, model_kwargs={}),
    dict(name='CIFAR', dataset_file='CIFAR', single_channel=False,
         method_cls=Method_CNN_CIFAR, model_kwargs={}),
]


def build_ablations(deep_channels, wide_hidden, stride_channels=None):
    stride_override = dict(stride=2)
    if stride_channels is not None:
        stride_override['conv_channels'] = stride_channels
    return [
        ('baseline',      {}),
        ('deeper-net',    dict(conv_channels=deep_channels)),   # depth
        ('wider-hidden',  dict(hidden_dim=wide_hidden)),        # hidden dim
        ('larger-kernel', dict(kernel_size=5, padding=2)),      # kernel size
        ('no-padding',    dict(padding=0)),                     # padding
        ('larger-stride', stride_override),                     # stride
        ('avg-pool',      dict(pool='avg')),                    # pooling: avg
        ('no-pool',       dict(pool='none')),                   # pooling: none
        ('mse-loss',      dict(loss_name='mse')),               # loss function
    ]
 
 
ABLATION_BASE = dict(
    conv_channels=[32, 64], hidden_dim=128, max_epoch=4, batch_size=128,
    optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    kernel_size=3, padding=1, stride=1, pool='max', pool_kernel=2,
)
ORL_ABLATION_BASE = dict(
    conv_channels=[16, 32], hidden_dim=128, max_epoch=10, batch_size=32,
    optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    kernel_size=3, padding=1, stride=1, pool='max', pool_kernel=2,
)
CIFAR_ABLATION_BASE = dict(
    conv_channels=[32, 64, 128], hidden_dim=256, max_epoch=5, batch_size=128,
    optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    kernel_size=3, padding=1, stride=1, pool='max', pool_kernel=2,
)
 
ABLATIONS = build_ablations([32, 64, 128], 256)
ORL_ABLATIONS = build_ablations([16, 32, 64], 256)
CIFAR_ABLATIONS = build_ablations([32, 64, 128, 256], 512,
                                  stride_channels=[32, 64])
 
 
def run_ablation_study(dataset_name, dataset_file, single_channel,
                       base_cfg, ablations):
    scores_by_name = {}
    method_cls = METHOD_CLASSES[dataset_name]
    print(f'\n===== {dataset_name} configuration impact study =====')
    for name, override in ablations:
        cfg = {**base_cfg, **override}
        run_name = f'{dataset_name}_{name}'
        scores_by_name[run_name] = run_experiment(
            name=run_name,
            dataset_file=dataset_file,
            single_channel=single_channel,
            model_kwargs=cfg,
            method_cls=method_cls,
            title=f'{dataset_name} Ablation - {name}',
        )
    return scores_by_name


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_scores = {}

    for run in RUNS:
        all_scores[run['name']] = run_experiment(**run)

    all_scores.update(run_ablation_study(
        'MNIST', 'MNIST', True, ABLATION_BASE, ABLATIONS,
    ))
    all_scores.update(run_ablation_study(
        'ORL', 'ORL', True, ORL_ABLATION_BASE, ORL_ABLATIONS,
    ))
    all_scores.update(run_ablation_study(
        'CIFAR', 'CIFAR', False, CIFAR_ABLATION_BASE, CIFAR_ABLATIONS,
    ))

    print('\n===== Final Summary =====')
    for k, v in all_scores.items():
        print(f'{k}: {format_scores(v)}')


if __name__ == '__main__':
    main()
