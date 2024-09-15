import torch
import numpy as np
import pandas as pd
import json



def data_transform_hier(traffic, user, n_his, n_pred, connect_nodes_traffic, connect_nodes_user, num_neighbors):
    n_route = traffic.shape[0]
    l = traffic.shape[1]
    num = (l - n_his - n_pred) // n_pred - 1

    x = torch.zeros([num, n_route, connect_nodes_traffic.shape[0], num_neighbors, n_his, 2])
    y = torch.zeros([num, n_route, n_pred, 2])
    traffic = torch.concat([traffic, torch.zeros([1, traffic.shape[1]])], dim = 0)
    user = torch.concat([user, torch.zeros([1, user.shape[1]])], dim = 0)
    # print(traffic.shape,user.shape)
    cnt = 0
    for i in range(num):
        head = n_pred * i
        tail = n_pred * i + n_his
        traffic_mx = traffic[:, head:tail]
        for j in range(len(connect_nodes_traffic)):
            # print(traffic_mx.shape,connect_nodes_traffic.shape)
            output_traffic = torch.index_select(traffic_mx, dim = 0, index = connect_nodes_traffic[j])
            # print(output_traffic.shape)
            output_traffic = output_traffic.reshape(n_route, num_neighbors, n_his, )
            x[cnt, :, j, :, :, 0] = output_traffic
            y[cnt, :, :, 0] = traffic[:-1, tail: tail + n_pred].reshape(-1, n_pred)
        for j in range(len(connect_nodes_user)):
            user_mx = user[:, head:tail]
            output_user = torch.index_select(user_mx, dim = 0, index = connect_nodes_user[j])
            # print(output_user.shape)
            output_user = output_user.reshape(n_route, num_neighbors, n_his, )
            x[cnt, :, j, :, :, 1] = output_user
            y[cnt, :, :, 1] = user[:-1, tail:tail + n_pred].reshape(-1, n_pred)
        cnt += 1

    x = x.reshape(num * n_route, connect_nodes_traffic.shape[0], num_neighbors, n_his, 2)
    y = y.reshape(num * n_route, n_pred, 2)
    return x, y

# traffic = torch.zeros([4505, 19])
# user = torch.zeros([4505, 19])

# with open('../data/base2hop_dis.json', 'r') as f:
#     base2hop = json.load(f)
# connect_nodes = torch.zeros([4, 4505 * 21], dtype = torch.int32)
# for base in base2hop:
#     hop_1 = torch.tensor(base2hop[base]['hop1'][:20])
#     idx = base2hop[base]['idx']
#     node_base = torch.cat([torch.tensor([idx]), hop_1, ], dim = 0)
#     connect_nodes[0, idx * 21:idx * 21 + 21] = node_base
#     connect_nodes[1, idx * 21:idx * 21 + 21] = node_base
#     connect_nodes[2, idx * 21:idx * 21 + 21] = node_base
#     connect_nodes[3, idx * 21:idx * 21 + 21] = node_base
# num_neighbors = 21
# x, y = data_transform_hier(traffic, user, 12, 1, connect_nodes, connect_nodes, num_neighbors)
# print(x.shape, y.shape)
