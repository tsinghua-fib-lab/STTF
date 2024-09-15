import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def evaluate_model_multitask(model, loss, data_iter, scaler_traffic, scaler_user, device, save = False):
    model.eval( )
    count = 0
    l_sum, n = 0.0, 0
    with torch.no_grad( ):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            # print(x.shape,y.shape,y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            y = y.detach( ).cpu( ).numpy( )
            y_pred = y_pred.detach( ).cpu( ).numpy( )

            if count == 0:
                data = y
                data_pred = y_pred
                count += 1
            else:
                # print(data.shape,data_pred.shape)
                data = np.concatenate((data, y), axis = 0)
                data_pred = np.concatenate((data_pred, y_pred), axis = 0)
        # print(data.shape,data_pred.shape)
        data = data.reshape(-1,4505,2)
        data_pred = data_pred.reshape(-1,4505,2)
        data[:, :, 0] = scaler_traffic.inverse_transform(data[:, :, 0])
        data_pred[:, :, 0] = scaler_traffic.inverse_transform(data_pred[:, :, 0])
        data[:, :, 1] = scaler_user.inverse_transform(data[:, :, 1])
        data_pred[:, :, 1] = scaler_user.inverse_transform(data_pred[:, :, 1])

        data = np.transpose(data, (1, 0, 2))
        data_pred = np.transpose(data_pred, (1, 0, 2))

        # xlabel = np.arange(data.shape[1])
        # plt.plot(xlabel,data[0,:,0],label = 'traffic')
        # plt.plot(xlabel, data[0, :, 1], label = 'user')
        # plt.plot(xlabel, data_pred[0, :, 0], label = 'traffic_pred')
        # plt.plot(xlabel, data_pred[0, :, 1], label = 'user_pred')
        # plt.legend()
        # plt.savefig('fig1.png')

        if save:
            test = pd.DataFrame(data = data)
            test.to_csv('traffic_true.csv', header = None, index = None)
            test = pd.DataFrame(data = data_pred)
            test.to_csv('traffic_pred.csv', header = None, index = None)

        d = np.abs(data[:, :, 0] - data_pred[:, :, 0])
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_traffic = np.array(mae).mean( )
        RMSE_traffic = np.sqrt(np.array(mse).mean( ))

        # 法1
        traffic = torch.tensor(data[:, :, 0]).reshape(-1, 1)
        traffic_pred = torch.tensor(data_pred[:, :, 0]).reshape(-1, 1)
        r2_traffic = 1 - torch.sum((traffic - traffic_pred) ** 2) / torch.sum((traffic - torch.mean(traffic)) ** 2)
        r2_traffic = r2_traffic.item( )

        d = np.abs(data[:, :, 1] - data_pred[:, :, 1])
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_user = np.array(mae).mean( )
        RMSE_user = np.sqrt(np.array(mse).mean( ))

        user = torch.tensor(data[:, :, 1]).reshape(-1, 1)
        user_pred = torch.tensor(data_pred[:, :, 1]).reshape(-1, 1)
        r2_user = 1 - torch.sum((user - user_pred) ** 2) / torch.sum((user - torch.mean(user)) ** 2)
        r2_user = r2_user.item( )

        # #法2
        # r2_list = []
        # for i in range(len(data)):
        #     r2_list.append(r2_score(data[i],data_pred[i]))
        # r2 = np.mean(np.array(r2_list))

        # #法3
        # data = torch.tensor(data)
        # data_pred = torch.tensor(data_pred)
        # r2_list = []
        # for i in range(len(data)):
        #     r2 = 1 - torch.sum((data[i] - data_pred[i]) ** 2) / torch.sum((data[i] - torch.mean(data[i])) ** 2)
        #     r2_list.append(r2.item())
        # r2 = np.mean(np.array(r2_list))

        # 法4
        # data = data.reshape(-1, 1)
        # data_pred = data_pred.reshape(-1, 1)
        # r2 = r2_score(data, data_pred)

        return l_sum / n, [MAE_traffic, MAE_user], [RMSE_traffic, RMSE_user], [r2_traffic, r2_user]


def evaluate_model_multitask_multistep(model, loss, data_iter, scaler_traffic, scaler_user, device, save = False):
    model.eval( )
    count = 0
    l_sum, n = 0.0, 0
    with torch.no_grad( ):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            # print(x.shape,y.shape,y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            y = y.detach( ).cpu( ).numpy( )
            y_pred = y_pred.detach( ).cpu( ).numpy( )

            y = y.reshape(-1,1,2)
            y_pred = y_pred.reshape(-1,1,2)

            if count == 0:
                data = y
                data_pred = y_pred
                count += 1
            else:
                # print(data.shape,data_pred.shape)
                data = np.concatenate((data, y), axis = 0)
                data_pred = np.concatenate((data_pred, y_pred), axis = 0)
        # print(data.shape,data_pred.shape)
        data = data.reshape(-1,4505,2)
        data_pred = data_pred.reshape(-1,4505,2)
        data[:, :, 0] = scaler_traffic.inverse_transform(data[:, :, 0])
        data_pred[:, :, 0] = scaler_traffic.inverse_transform(data_pred[:, :, 0])
        data[:, :, 1] = scaler_user.inverse_transform(data[:, :, 1])
        data_pred[:, :, 1] = scaler_user.inverse_transform(data_pred[:, :, 1])

        data = np.transpose(data, (1, 0, 2))
        data_pred = np.transpose(data_pred, (1, 0, 2))

        # xlabel = np.arange(data.shape[1])
        # plt.plot(xlabel,data[0,:,0],label = 'traffic')
        # plt.plot(xlabel, data[0, :, 1], label = 'user')
        # plt.plot(xlabel, data_pred[0, :, 0], label = 'traffic_pred')
        # plt.plot(xlabel, data_pred[0, :, 1], label = 'user_pred')
        # plt.legend()
        # plt.savefig('fig1.png')

        if save:
            test = pd.DataFrame(data = data)
            test.to_csv('traffic_true.csv', header = None, index = None)
            test = pd.DataFrame(data = data_pred)
            test.to_csv('traffic_pred.csv', header = None, index = None)

        d = np.abs(data[:, :, 0] - data_pred[:, :, 0])
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_traffic = np.array(mae).mean( )
        RMSE_traffic = np.sqrt(np.array(mse).mean( ))

        # 法1
        traffic = torch.tensor(data[:, :, 0]).reshape(-1, 1)
        traffic_pred = torch.tensor(data_pred[:, :, 0]).reshape(-1, 1)
        r2_traffic = 1 - torch.sum((traffic - traffic_pred) ** 2) / torch.sum((traffic - torch.mean(traffic)) ** 2)
        r2_traffic = r2_traffic.item( )

        d = np.abs(data[:, :, 1] - data_pred[:, :, 1])
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_user = np.array(mae).mean( )
        RMSE_user = np.sqrt(np.array(mse).mean( ))

        user = torch.tensor(data[:, :, 1]).reshape(-1, 1)
        user_pred = torch.tensor(data_pred[:, :, 1]).reshape(-1, 1)
        r2_user = 1 - torch.sum((user - user_pred) ** 2) / torch.sum((user - torch.mean(user)) ** 2)
        r2_user = r2_user.item( )

        # #法2
        # r2_list = []
        # for i in range(len(data)):
        #     r2_list.append(r2_score(data[i],data_pred[i]))
        # r2 = np.mean(np.array(r2_list))

        # #法3
        # data = torch.tensor(data)
        # data_pred = torch.tensor(data_pred)
        # r2_list = []
        # for i in range(len(data)):
        #     r2 = 1 - torch.sum((data[i] - data_pred[i]) ** 2) / torch.sum((data[i] - torch.mean(data[i])) ** 2)
        #     r2_list.append(r2.item())
        # r2 = np.mean(np.array(r2_list))

        # 法4
        # data = data.reshape(-1, 1)
        # data_pred = data_pred.reshape(-1, 1)
        # r2 = r2_score(data, data_pred)

        return l_sum / n, [MAE_traffic, MAE_user], [RMSE_traffic, RMSE_user], [r2_traffic, r2_user]



def evaluate_model(model, loss, data_iter, scaler_traffic, scaler_user, device, save = False):
    model.eval( )
    count = 0
    l_sum, n = 0.0, 0
    with torch.no_grad( ):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            y_pred = model(x).squeeze(-1)
            # print(x.shape,y.shape,y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            y = y.detach( ).cpu( ).numpy( )
            y_pred = y_pred.detach( ).cpu( ).numpy( )

            if count == 0:
                data = y
                data_pred = y_pred
                count += 1
            else:
                # print(data.shape,data_pred.shape)
                data = np.concatenate((data, y), axis = 0)
                data_pred = np.concatenate((data_pred, y_pred), axis = 0)

        # print(data.shape,data_pred.shape)
        data = scaler_traffic.inverse_transform(data)
        data_pred = scaler_traffic.inverse_transform(data_pred)
        # data[:, :, 1] = scaler_user.inverse_transform(data[:, :, 1])
        # data_pred[:, :, 1] = scaler_user.inverse_transform(data_pred[:, :, 1])

        data = np.transpose(data, (1, 0))
        data_pred = np.transpose(data_pred, (1, 0))

        # xlabel = np.arange(data.shape[1])
        # plt.plot(xlabel,data[0,:,0],label = 'traffic')
        # plt.plot(xlabel, data[0, :, 1], label = 'user')
        # plt.plot(xlabel, data_pred[0, :, 0], label = 'traffic_pred')
        # plt.plot(xlabel, data_pred[0, :, 1], label = 'user_pred')
        # plt.legend()
        # plt.savefig('fig1.png')

        if save:
            test = pd.DataFrame(data = data)
            test.to_csv('traffic_true.csv', header = None, index = None)
            test = pd.DataFrame(data = data_pred)
            test.to_csv('traffic_pred.csv', header = None, index = None)

        d = np.abs(data - data_pred)
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_traffic = np.array(mae).mean( )
        RMSE_traffic = np.sqrt(np.array(mse).mean( ))

        # 法1
        data = torch.tensor(data).reshape(-1, 1)
        data_pred = torch.tensor(data_pred).reshape(-1, 1)
        r2 = 1 - torch.sum((data - data_pred) ** 2) / torch.sum((data - torch.mean(data)) ** 2)
        r2 = r2.item( )

        # #法2
        # r2_list = []
        # for i in range(len(data)):
        #     r2_list.append(r2_score(data[i],data_pred[i]))
        # r2 = np.mean(np.array(r2_list))

        # #法3
        # data = torch.tensor(data)
        # data_pred = torch.tensor(data_pred)
        # r2_list = []
        # for i in range(len(data)):
        #     r2 = 1 - torch.sum((data[i] - data_pred[i]) ** 2) / torch.sum((data[i] - torch.mean(data[i])) ** 2)
        #     r2_list.append(r2.item())
        # r2 = np.mean(np.array(r2_list))

        # 法4
        data = data.reshape(-1, 1)
        data_pred = data_pred.reshape(-1, 1)
        r2 = r2_score(data, data_pred)

        return l_sum / n, MAE_traffic, RMSE_traffic, r2


def evaluate_model_decomp(model, loss, data_iter, scaler_traffic, scaler_user, device, idx, save = False):
    model.eval( )
    count = 0
    l_sum, n = 0.0, 0
    with torch.no_grad( ):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device).squeeze( )
            x = x[:, :, :, idx]
            y = y[:, :, idx]
            y_pred = model(x).squeeze( )
            # print(x.shape,y.shape,y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            y = y.detach( ).cpu( ).numpy( )
            y_pred = y_pred.detach( ).cpu( ).numpy( )

            if count == 0:
                data = y
                data_pred = y_pred
                count += 1
            else:
                # print(data.shape,data_pred.shape)
                data = np.concatenate((data, y), axis = 0)
                data_pred = np.concatenate((data_pred, y_pred), axis = 0)
        if idx == 0:
            data = scaler_traffic.inverse_transform(data)
            data_pred = scaler_traffic.inverse_transform(data_pred)
        if idx == 1:
            data = scaler_user.inverse_transform(data)
            data_pred = scaler_user.inverse_transform(data_pred)

        data = np.transpose(data, (1, 0))
        data_pred = np.transpose(data_pred, (1, 0))

        # xlabel = np.arange(data.shape[1])
        # plt.plot(xlabel,data[0,:,0],label = 'traffic')
        # plt.plot(xlabel, data[0, :, 1], label = 'user')
        # plt.plot(xlabel, data_pred[0, :, 0], label = 'traffic_pred')
        # plt.plot(xlabel, data_pred[0, :, 1], label = 'user_pred')
        # plt.legend()
        # plt.savefig('fig1.png')

        traffic = data

        traffic_pred = data_pred

        if save:
            test = pd.DataFrame(data = traffic)
            test.to_csv('traffic_true.csv', header = None, index = None)
            test = pd.DataFrame(data = traffic_pred)
            test.to_csv('traffic_pred.csv', header = None, index = None)

        d = np.abs(traffic - traffic_pred)
        mae = d.tolist( )
        mse = (d ** 2).tolist( )
        MAE_traffic = np.array(mae).mean( )
        RMSE_traffic = np.sqrt(np.array(mse).mean( ))
        data = torch.tensor(data)
        data_pred = torch.tensor(data_pred)
        r2_traffic = 1 - torch.sum((data - data_pred) ** 2) / torch.sum((data - torch.mean(data)) ** 2)

        return l_sum / n, MAE_traffic, RMSE_traffic, r2_traffic.item( )

