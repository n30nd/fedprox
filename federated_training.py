import torch
from model import ResNet18, VGG11Model
import torch.optim as optim 
import copy
import random 
import numpy as np
import time 
import matplotlib.pyplot as plt
from data_utils import get_val_dataloader
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def reset_model_to_zero(model):
    for param in model.parameters():
        param.data.fill_(0.0)

def federated_train(trainloaders, valloaders, testloader, config):
    #Cai dat seed
    random.seed(config.dataset_seed)
    np.random.seed(config.dataset_seed)
    torch.manual_seed(config.dataset_seed)
    torch.cuda.manual_seed(config.dataset_seed)
    torch.cuda.manual_seed_all(config.dataset_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model = ResNet18(num_classes=2)
    model = VGG11Model(num_classes=2)
    nets = {net_i: copy.deepcopy(model) for net_i in range(len(trainloaders))}
    global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
    valloader_goc = get_val_dataloader()

    num_rounds = config.num_rounds  # Số vòng huấn luyện
    accs_test = []
    accs_val = []
    accs_test.append(evaluate(global_model, testloader))
    accs_val.append(evaluate(global_model, valloader_goc))
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        start = time.time()
        global_para = global_model.state_dict()

        # Chọn các client tham gia vào mỗi round
        selected_clients = select_clients(trainloaders, config.clients_per_round)
        
        # Huấn luyện trên các client đã chọn
        for client in selected_clients:
            nets[client].load_state_dict(global_para)
        local_train_net_fedprox(nets, selected_clients, global_model, config, trainloaders, device=DEVICE)

        total_data_points = sum([len(trainloaders[client].dataset) for client in selected_clients])
        freqs = [len(trainloaders[client].dataset) / total_data_points for client in selected_clients]

        for idx in range(len(selected_clients)):
            net_para = nets[selected_clients[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * freqs[idx]
        global_model.load_state_dict(global_para)
        global_model.to('cpu')
        acc_test = evaluate(global_model, testloader)        
        accs_test.append(acc_test)

        acc_val = evaluate(global_model, valloader_goc)
        accs_val.append(acc_val)

        print(f"Round {round_num + 1} Test Accuracy: {acc_test:.2f}%")
        print(f"Round {round_num + 1} Val Accuracy: {acc_val:.2f}%")

        if round_num >= 0:
            if acc_val > 80.0:
                config.learning_rate = 1e-8
                print(f"Accuracy > 80%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 70.0:
                config.learning_rate = 1e-7
                print(f"Accuracy > 70%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 60.0:
                config.learning_rate = 1e-6
                print(f"Accuracy > 60%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 50.0:
                config.learning_rate = 1e-5
                print(f"Accuracy > 50%, decreasing learning rate to {config.learning_rate}")
            else :
                config.learning_rate = 1e-4
                print(f"Accuracy <= 50%, increasing learning rate to {config.learning_rate}")

        end = time.time()
        print(f'Time for round {round_num + 1}: ', end-start)
    # plot_accuracy(accs)
    print('accuracies test: ', accs_test)
    print('accuracies val: ', accs_val)
    plt.plot(range(0, num_rounds + 1), accs_test, marker='o', label='Accuracy_test')
    plt.plot(range(0, num_rounds + 1), accs_val, marker='x', label='Accuracy_val')
    plt.xlabel('Round')
    plt.xticks(range(0, num_rounds + 1, 10))
    plt.ylabel('Accuracy')
    plt.title('WOW WOW WOW')
    plt.grid(True)
    plt.legend()
    plt.savefig('running_outputs/accuracy_summary.png')
    plt.close()



def local_train_net_fedprox(nets, selected_clients, global_model, config, trainloaders, device='cpu'):
    for net_id in selected_clients:
        net = nets[net_id]
        net.to(device)
        train_net_fedprox(net, global_model, trainloaders[net_id], config, device=device)


def train_net_fedprox(net, global_model, trainloader, config, device):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.learning_rate,
        momentum=config.momentum
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)
    global_weights = list(global_model.to(device).parameters())
    mu = config.mu
    for _ in range(config.num_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)

            # Tính toán phần regularization
            reg_loss = 0.0
            for param_index, param in enumerate(net.parameters()):
                reg_loss += ((mu / 2) * torch.norm((param - global_weights[param_index])) ** 2)
            loss += reg_loss
            
            loss.backward()
            optimizer.step()
    net.to('cpu')
      
        



def select_clients(trainloaders, clients_per_round):
    """Chọn ngẫu nhiên một số client tham gia huấn luyện trong mỗi round."""
    # Số lượng client có sẵn
    total_clients = len(trainloaders)
    # Chọn ngẫu nhiên một số client
    selected_clients = random.sample(range(total_clients), clients_per_round)
    return selected_clients


def evaluate(model, testloader):
    """Đánh giá mô hình trên tập kiểm tra."""
    model.eval()  # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
