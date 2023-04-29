import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

import utils

class SimpleDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class NotearsPerm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.A = nn.Parameter(torch.randn(dim, dim, requires_grad=True))
        self.P = nn.Parameter(torch.abs(torch.randn(dim, dim, requires_grad=True)))

    def forward(self, X):
        """
        X: (N, dim)
        """
        X_perm = torch.matmul(self.P, X.T).T
        A_upper_tri = torch.triu(self.A, diagonal=1)
        X_new = torch.matmul(A_upper_tri.T, X_perm.T).T
        return X_new
    
def perm_loss_func(M):
    n = M.shape[0]
    p_loss = 0
    for i in range(n):
        a = torch.sum(torch.abs(M[i]))
        b = torch.sum(M[i] ** 2) ** 0.5
        p_loss += a - b
    for j in range(n):
        a = torch.sum(torch.abs(M[:, j]))
        b = torch.sum(M[:, j] ** 2) ** 0.5
        p_loss += a - b
    return p_loss

def permutation_matrix(M):
    P = np.zeros_like(M)
    d = M.shape[0]

    val = []
    index = []
    for i in range(d):
        for j in range(d):
            val.append(M[i,j])
            index.append((i, j))
    val = np.asarray(val)
    index = np.asarray(index)
    descent_index = index[np.argsort(val)[::-1]]

    r_set = set()
    c_set = set()
    for r,c in descent_index:
        if r in r_set or c in c_set:
            continue
        P[r, c] = 1
        r_set.add(r)
        c_set.add(c)
    return P


def trainer(data):
    EPOCHS = 400
    bs = 20

    data = torch.from_numpy(data)
    data = data - data.mean(axis=0, keepdims=True)
    dim = data.shape[-1]
    train_dataset = SimpleDataset(data)
    dataloader = DataLoader(train_dataset, batch_size=bs, drop_last=False, shuffle=True, num_workers=2)

    model = NotearsPerm(dim).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # import pdb; pdb.set_trace()
    for ep in range(EPOCHS):
        for idx, x in enumerate(dataloader):
            model.train()
            input = x.cuda().float()
            out = model(input)

            perm_input = torch.matmul(model.P, input.T).T
            l2_loss = torch.sum((out - perm_input) ** 2) / input.shape[0]
            reg_loss = torch.sum(torch.abs(torch.triu(model.A, diagonal=1)))
            perm_loss = perm_loss_func(model.P)

            loss = l2_loss + reg_loss + perm_loss
            if idx % 40 == 0:
                print('{} \t {} \t {} \t {} \t {} \t {}'.format(ep, idx, l2_loss.item(), reg_loss.item(), perm_loss.item(), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                p_next = model.P.data.clone().detach()
                p_next = torch.clamp(p_next, min=0)
                p_next = p_next / p_next.sum(dim=1, keepdim=True)
                p_next = p_next / p_next.sum(dim=0, keepdim=True)
                model.P.data = p_next            

    P_final = model.P.clone().detach().cpu().numpy()
    P_final = permutation_matrix(P_final)
    A_final = torch.triu(model.A, diagonal=1).detach().cpu().numpy()
    print(P_final)
    print(A_final)
    w_est = P_final.T @ A_final @ P_final
    return w_est



if __name__ == '__main__':
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 1000, 50, 50, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)

    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    with open('graph_{}_{}.pkl'.format(d, s0), 'wb') as f:
        graph_out = {'B': B_true, 'W': W_true, 'X': X}
        pickle.dump(graph_out, f)

    with open('graph_50_50.pkl', 'rb') as f:
        graph_data = pickle.load(f)
        B_true = graph_data['B']
        W_true = graph_data['W']
        X = graph_data['X']

    W_pred = trainer(X)
    
    with open('perm_linear_pred.pkl', 'wb') as f:
        pickle.dump(W_pred, f)
    
    with open('perm_linear_pred.pkl', 'rb') as f:
        W_ins = pickle.load(f)
    W_ins[np.abs(W_ins) < 0.3] = 0
    print(W_ins)
    assert utils.is_dag(W_ins)
    np.savetxt('W_est.csv', W_ins, delimiter=',')
    acc = utils.count_accuracy(B_true, W_ins != 0)
    print(acc)

