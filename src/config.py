import yaml
from pathlib import Path
import torch

CONFIG = {
    'data': {
        'train_image_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/train/images',
        'train_mask_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/train/masks',
        'val_image_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/val/images',
        'val_mask_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/val/masks',
        'test_image_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/test/images',
        'test_mask_dir': 'F:/projects/hirdl/FedSSL_Paper/data_set/kits_2d_splitted/test/masks',
        'image_size': 224,
        'batch_size': 32,
    },
    'model': {
        'backbone': 'resnet18',
        'num_classes': 1,  # Binary detection
    },
    'training': {
        'epochs': 20,
        'lr': 0.001,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    },
    'federated': {
        'num_clients': 5,
        'local_epochs': 3,
    },
    'outputs': {
        'checkpoints': 'outputs/checkpoints',
        'logs': 'outputs/logs',
    }
}

# Save/load config
def save_config(file='config.yaml'):
    with open(file, 'w') as f:
        yaml.dump(CONFIG, f)

save_config()