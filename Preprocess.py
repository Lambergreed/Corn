import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer

def read_data():
    corn = np.genfromtxt('edit\gwp_ns_2000_2008.csv', skip_header=1, delimiter=',')
    # imp = Imputer(missing_values='NAN', strategy='mean', axis=0, verbose=0, copy=True)
    # corn = imp.fit_transform(corn)
    [m, n] = corn.shape
    tr_index = []
    tes_index = []
    random.seed(1)
    te_index = random.sample(range(m), int(m/10))

    for j in range(m):
        if j in te_index:
            tes_index.append(j)
        else:
            tr_index.append(j)
    tr_data = corn[tr_index, 1:]
    te_data = corn[te_index, 1:]

    # Normalization
    tr_data = scale(tr_data, axis=0)
    te_data = scale(te_data, axis=0)
    # tr_data = normalize(tr_data, axis=0)
    # te_data = normalize(te_data, axis=0)

    tr_label = corn[tr_index, 0]
    te_label = corn[te_index, 0]
    tr_label = np.reshape(tr_label, [-1, 1])
    te_label = np.reshape(te_label, [-1, 1])

    return tr_data, tr_label, te_data, te_label

if __name__ == '__main__':
    read_data()