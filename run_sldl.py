import os
import numpy as np
from methods.sldl import SLDL
from utils.utils import set_seed, evaluation_xmc, get_inv_propesity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import argparse


def get_model():
    model = SLDL(num_classes=y_train.shape[1],
                 num_feature=x_train.shape[1],
                 label_embedding_dim=label_embedding_dim,
                 label_embedding_threshold=label_embedding_threshold,
                 knn_neighbors=knn_neighbors,
                 method=method,
                 device=device)
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sample_data')
parser.add_argument('--method', type=str, default='ridge-lbfgs')
parser.add_argument('--label_embedding_threshold', type=float, default=0.1)
parser.add_argument('--label_embedding_dim', type=int, default=128)
parser.add_argument('--knn_neighbors', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    algorithm = 'sldl'
    dataset_name = args.dataset_name
    method = args.method
    label_embedding_threshold = args.label_embedding_threshold
    label_embedding_dim = args.label_embedding_dim
    knn_neighbors = args.knn_neighbors
    device = args.device
    seed = args.seed
    set_seed(seed)

    x_train = normalize(np.load(f'data/{dataset_name}_x_train.npy'))
    x_test = normalize(np.load(f'data/{dataset_name}_x_test.npy'))
    y_train = np.load(f'data/{dataset_name}_y_train.npy')
    y_test = np.load(f'data/{dataset_name}_y_test.npy')

    result_path = os.path.join('save', f'{algorithm}_{method}',
                               f'{label_embedding_dim}_{label_embedding_threshold}_{seed}')

    if os.path.exists(os.path.join(result_path, f'{dataset_name}_{knn_neighbors}_y_prob.npy')):
        print(algorithm, dataset_name, method, 'exist!')
        y_prob = np.load(os.path.join(result_path,
                                      f'{dataset_name}_{knn_neighbors}_y_prob.npy'))
    else:
        os.makedirs(result_path, exist_ok=True)
        print(algorithm, dataset_name, method, seed, 'Training!')
        model = get_model()
        if os.path.exists(os.path.join(result_path, f'{dataset_name}_Z.npy')):
            Z = np.load(os.path.join(result_path, f'{dataset_name}_Z.npy'))
        else:
            Z = model.get_embedding(y_train)
            np.save(os.path.join(result_path, f'{dataset_name}_Z.npy'), Z)
        model.train(x_train, Z)
        y_prob = model.predict(x_train, y_train, x_test)
        np.save(os.path.join(result_path, f'{dataset_name}_{knn_neighbors}_y_prob.npy'), y_prob)
    result = evaluation_xmc(csr_matrix(y_test), csr_matrix(y_prob),
                            get_inv_propesity(dataset_name, csr_matrix(y_train)))
    for key in result.keys():
        print(f'{key}: {result[key]}')
