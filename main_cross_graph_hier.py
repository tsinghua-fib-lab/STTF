# mlflow ui --backend-store-uri='./mlflow_output' --port=port_number
# torch-1.10-py3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import mlflow
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from load_data import *
from utils import *
from transformer import *
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from mlflow.tracking import MlflowClient
import warnings
import setproctitle


warnings.filterwarnings('ignore')

seed = 13
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

name = 'cross_graph_hier'

setproctitle.setproctitle(name + "@gongjh")  #################################################
device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
# device_ids = [0,1,2,3]

traffic_filename = '../data/traffic.csv'
traffic_mx = pd.read_csv(traffic_filename, header = None)
traffic_mx = np.log10(traffic_mx.values)

user_filename = '../data/user.csv'
user_mx = pd.read_csv(user_filename, header = None)
# user_mx = np.log10(user_mx.values)
user_mx = user_mx.values

num_neighbors = 21
connect_nodes_traffic = torch.zeros((4, traffic_mx.shape[0] * num_neighbors), dtype = torch.int32)
connect_nodes_user = torch.zeros((4, traffic_mx.shape[0] * num_neighbors), dtype = torch.int32)

with open('../data/base2hop_dis.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_traffic[0, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base
    connect_nodes_user[0, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base

with open('../data/base2hop_poi.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_traffic[1, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base
    connect_nodes_user[1, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base

with open('../data/base2hop_pattern.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_traffic[2, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base

with open('../data/base2hop_user_pattern.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_user[2, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base

with open('../data/base2hop_dtw.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_traffic[3, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base

with open('../data/base2hop_user_dtw.json', 'r') as f:
    base2hop = json.load(f)
for base in base2hop:
    hop_1 = torch.tensor(base2hop[base]['hop1'][:num_neighbors - 1])
    idx = base2hop[base]['idx']
    node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
    connect_nodes_user[3, idx * num_neighbors:idx * num_neighbors + num_neighbors] = node_base


split_line1 = int(traffic_mx.shape[1] * 0.7)
split_line2 = int(traffic_mx.shape[1] * 0.85)

traffic_train = traffic_mx[:, :split_line1]
traffic_val = traffic_mx[:, split_line1:split_line2]
traffic_test = traffic_mx[:, split_line2:]

user_train = user_mx[:, :split_line1]
user_val = user_mx[:, split_line1:split_line2]
user_test = user_mx[:, split_line2:]
# print(user_test.shape)
traffic_train = traffic_train.T
traffic_val = traffic_val.T
traffic_test = traffic_test.T
scaler_traffic = StandardScaler( )
traffic_train = scaler_traffic.fit_transform(traffic_train)
traffic_val = scaler_traffic.transform(traffic_val)
traffic_test = scaler_traffic.transform(traffic_test)
traffic_train = traffic_train.T
traffic_val = traffic_val.T
traffic_test = traffic_test.T

user_train = user_train.T
user_val = user_val.T
user_test = user_test.T
scaler_user = StandardScaler( )
user_train = scaler_user.fit_transform(user_train)
user_val = scaler_user.transform(user_val)
user_test = scaler_user.transform(user_test)
user_train = user_train.T
user_val = user_val.T
user_test = user_test.T

n_nodes = traffic_mx.shape[0]
n_his = 12
n_pred = 1
lr = 0.001
epochs = 50
batch_size = 8
traffic_train, user_train = torch.tensor(traffic_train), torch.tensor(user_train)
traffic_val, user_val = torch.tensor(traffic_val), torch.tensor(user_val)
traffic_test, user_test = torch.tensor(traffic_test), torch.tensor(user_test)

x_train, y_train = data_transform_hier(traffic_train, user_train, n_his, n_pred, connect_nodes_traffic,
                                       connect_nodes_user, num_neighbors)
x_val, y_val = data_transform_hier(traffic_val, user_val, n_his, n_pred, connect_nodes_traffic, connect_nodes_user,
                                   num_neighbors)
x_test, y_test = data_transform_hier(traffic_test, user_test, n_his, n_pred, connect_nodes_traffic, connect_nodes_user,
                                     num_neighbors)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

experiment_name = name  ##############################################
mlflow.set_tracking_uri('./mlflow_output')
client = MlflowClient( )

try:
    EXP_ID = client.create_experiment(experiment_name)
    print('Initial Create!', EXP_ID)
except:
    experiments = client.get_experiment_by_name(experiment_name)
    EXP_ID = experiments.experiment_id
    print('Experiment Exists, Continuing', EXP_ID)

with mlflow.start_run(experiment_id = EXP_ID):
    archive_path = mlflow.get_artifact_uri( )
    params = {'dataset': 'shanghai'}  ################################################
    mlflow.log_params(params)

    save_path = 'models/' + name + '.pt'  ############################################
    loss = nn.MSELoss( ).to(device)
    model = Cross_Graph_hier( ).to(device)  ###########################################
    # model = nn.DataParallel(model,device_ids)
    optimize = torch.optim.Adam(model.parameters( ), lr)
    min_val_loss = np.inf
    best_val_it = 0
    for epoch in range(1, epochs):
        start_train = time.time( )
        print('epoch:', epoch)
        total_train_step = 0
        l_sum, n = 0.0, 0
        model.train( )

        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            # print(x.shape,y.shape)
            y_pred = model(x)
            # print(x.shape,y.shape,y_pred.shape)
            l = loss(y_pred, y)
            optimize.zero_grad( )
            l.backward( )
            optimize.step( )
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            total_train_step += 1

        val_loss, MAE, RMSE, R2 = evaluate_model_multitask(model, loss, val_iter, scaler_traffic, scaler_user, device,
                                                           )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict( ), save_path)
            best_val_it = epoch
        mlflow.log_metrics(
            {'train_time': time.time( ) - start_train, 'train_loss': l_sum / n, 'validation loss': val_loss,
             'best_epoch': best_val_it,
             'MAE_traffic': MAE[0], 'RMSE_traffic': RMSE[0], 'R2_traffic': R2[0], 'MAE_user': MAE[1],
             'RMSE_user': RMSE[1], 'R2_user': R2[1], }, step = epoch)

    best_model = Cross_Graph_hier( ).to(device)  ####################################################################
    best_model.load_state_dict(torch.load(save_path))
    test_loss, MAE, RMSE, R2 = evaluate_model_multitask(best_model, loss, test_iter, scaler_traffic, scaler_user,
                                                        device)
    print("test loss:", test_loss, "\nMAE:", MAE, ", RMSE:", RMSE, ", R2:", R2)
    mlflow.log_params({'test_loss': test_loss, 'MAE_traffic': MAE[0], 'RMSE_traffic': RMSE[0], 'R2_traffic': R2[0],
                       'MAE_user': MAE[1], 'RMSE_user': RMSE[1], 'R2_user': R2[1], })

save_path = 'models/' + name + '.pt'  ############################################
loss = nn.MSELoss( ).to(device)
best_model = Cross_Graph_hier( ).to(device)  ####################################################################
# best_model = nn.DataParallel(best_model,device_ids)
best_model.load_state_dict(torch.load(save_path))
# best_model = best_model.module
test_loss, MAE, RMSE, R2 = evaluate_model_multitask(best_model, loss, test_iter, scaler_traffic, scaler_user,
                                                    device)
print("test loss:", test_loss, "\nMAE:", MAE, ", RMSE:", RMSE, ", R2:", R2)
mlflow.log_params({'test_loss': test_loss, 'MAE_traffic': MAE[0], 'RMSE_traffic': RMSE[0], 'R2_traffic': R2[0],
                    'MAE_user': MAE[1], 'RMSE_user': RMSE[1], 'R2_user': R2[1], })