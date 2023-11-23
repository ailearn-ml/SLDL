import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import numpy as np
import copy
from tqdm import tqdm


def batch_kl_normal(pm, pv):
    qm = torch.unsqueeze(pm, dim=1)
    qv = torch.unsqueeze(pv, dim=1)
    pm = torch.unsqueeze(pm, dim=0)
    pv = torch.unsqueeze(pv, dim=0)
    element_wise = 0.5 * (torch.log(qv) - torch.log(pv) + pv / qv + (pm - qm).pow(2) / qv - 1)
    kl = element_wise.sum(-1)
    return kl


class SLDL:
    def __init__(self, num_classes, num_feature, label_embedding_dim=32, lr=0.001,
                 label_embedding_threshold=0., knn_neighbors=30, method='ridge-lbfgs', device='cuda:0'):
        super(SLDL, self).__init__()
        self.num_classes = num_classes
        self.num_feature = num_feature
        self.knn_neighbors = knn_neighbors
        self.label_embedding_dim = label_embedding_dim
        self.label_embedding_threshold = label_embedding_threshold
        self.method = method
        if self.method == 'ridge':
            self.regressor = Ridge()
        elif self.method == 'ridge-saga':
            self.regressor = Ridge(solver='saga')
        elif self.method == 'ridge-lbfgs':
            self.regressor = Ridge(solver='lbfgs', positive=True)
        else:
            raise ValueError('Wrong Method!')
        self.lr = lr
        self.device = device

    def to_tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def get_embedding(self, y_train):
        graph = np.zeros([y_train.shape[1], y_train.shape[1]])
        for i in tqdm(range(y_train.shape[0])):
            pos = np.arange(y_train.shape[1])[y_train[i] == 1]
            for p in pos:
                graph[p, pos] += 1
        LG = np.zeros_like(graph)
        LG[graph > 0] = 1
        LG = LG / np.sum(LG, axis=1, keepdims=True)
        p = copy.deepcopy(LG)
        p_total = copy.deepcopy(p)
        gamma = 1
        # -------------------------------- Walk 10 times --------------------------------
        for i in range(10):
            p = p @ LG
            gamma *= 0.5
            p_total += gamma * p
        p_total = p_total / np.sum(p_total, axis=1, keepdims=True)

        idx = []
        for i in range(p_total.shape[0]):
            max_idx = np.argsort(p_total[i])[::-1]
            idx.append(max_idx)
        idx = torch.from_numpy(np.array(idx))
        mu = torch.rand([y_train.shape[1], self.label_embedding_dim]).to(self.device).requires_grad_(True)
        sigma = torch.rand([y_train.shape[1], self.label_embedding_dim]).to(self.device).requires_grad_(True)
        mu.data.normal_(0, 0.1)
        sigma.data.uniform_(0.0, 0.5)
        optimizer = torch.optim.AdamW([mu, sigma], lr=self.lr)
        for _ in tqdm(range(50)):
            kl = batch_kl_normal(mu, sigma)
            loss = []
            for i in range(idx.shape[0]):
                kl_sorted = kl[i, idx[i]]
                loss.append((torch.relu(kl_sorted[1:] + self.label_embedding_threshold - kl_sorted[:-1])).sum())
            loss_mean = sum(loss) / len(loss)
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
        mu = mu.detach().cpu().numpy()
        Z = y_train @ mu
        return Z

    def train(self, x_train, Z):
        self.regressor.fit(x_train, Z)

    def predict(self, x_train, y_train, x_test):
        z_train = self.regressor.predict(x_train)
        z_test = self.regressor.predict(x_test)
        model = KNeighborsRegressor(n_neighbors=self.knn_neighbors, weights='distance', metric='cosine')
        model.fit(z_train, y_train)
        y_pred = model.predict(z_test)
        return y_pred
