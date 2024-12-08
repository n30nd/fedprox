# import argparse
# from federated_training import federated_train
# from data_utils import load_datasets
# import yaml

# def parse_args():
#     parser = argparse.ArgumentParser(description='Federated Learning with FedNova and ResNet18')
#     parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
#     parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
#     parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
#     parser.add_argument('--clients_per_round', type=int, default=2, help='Number of clients selected per round')
#     parser.add_argument('--fraction_fit', type=float, default=0.1, help='Fraction of clients to fit')
#     parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
#     parser.add_argument('--num_rounds', type=int, default=2, help='Number of communication rounds')
#     parser.add_argument('--partitioning', type=str, default="imbalance_label", help='Type of data partitioning')
#     parser.add_argument('--dataset_seed', type=int, default=42, help='Seed for dataset partitioning')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet parameter for imbalance')
#     parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for noise-based imbalance')
#     parser.add_argument('--labels_per_client', type=int, default=2, help='Labels per client in partitioning')
#     parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimization')
#     parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
#     return parser.parse_args()

# def load_config(config_file):
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# def main():
#     # Parse arguments
#     args = parse_args()

#     # Load configuration file
#     config = load_config('config.yaml')
#     print(config)
#     # Load dataset
#     trainloaders, valloaders, testloader = load_datasets(config.dataset, args.num_clients)

#     # Train federated model
#     federated_train(trainloaders, valloaders, testloader, config, args)

# if __name__ == '__main__':
#     main()


import argparse
from federated_training import federated_train
from data_utils import load_datasets
from omegaconf import OmegaConf  # Thêm OmegaConf
import os

# def parse_args():
#     parser = argparse.ArgumentParser(description='Federated Learning with FedNova and ResNet18')
#     parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
#     parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
#     parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
#     parser.add_argument('--clients_per_round', type=int, default=2, help='Number of clients selected per round')
#     parser.add_argument('--fraction_fit', type=float, default=0.1, help='Fraction of clients to fit')
#     parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
#     parser.add_argument('--num_rounds', type=int, default=2, help='Number of communication rounds')
#     parser.add_argument('--partitioning', type=str, default="imbalance_label", help='Type of data partitioning')
#     parser.add_argument('--dataset_seed', type=int, default=42, help='Seed for dataset partitioning')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet parameter for imbalance')
#     parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for noise-based imbalance')
#     parser.add_argument('--labels_per_client', type=int, default=2, help='Labels per client in partitioning')
#     parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimization')
#     parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
#     return parser.parse_args()

def load_config(config_file):
    """Load configuration using OmegaConf (DictConfig)"""
    # Dùng OmegaConf để load file config.yaml
    config = OmegaConf.load(config_file)
    return config

def main():
    # Parse arguments
    # args = parse_args()

    # Kiểm tra file config.yaml có tồn tại không
    config_file = 'config.yaml'
    if not os.path.exists(config_file):
        print(f"Config file '{config_file}' not found. Please provide a valid config file.")
        return

    # Load configuration file
    config = load_config(config_file)  # Trả về DictConfig

    # Kiểm tra các tham số được thay thế chính xác
    print("Loaded Config:")
    print(config)

    # Load dataset
    # trainloaders, valloaders, testloader = load_datasets(config.dataset.name, args.num_clients)
    trainloaders, valloaders, testloader = load_datasets(
        config=config.dataset,
        num_clients=config.num_clients,
        val_ratio=config.dataset.val_split,
    )

    # Train federated model
    federated_train(trainloaders, valloaders, testloader, config)

if __name__ == '__main__':
    main()
