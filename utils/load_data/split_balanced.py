import numpy as np

def split_balanced(data, target, test_size=0.2):

    classes, class_counts = np.unique(target, return_counts=True)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    # n_train = max(0, len(target)-n_test)
    # n_train_per_class = max(1, int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1, n_test)
    n_train_per_class = class_counts - n_test_per_class

    index_train = []
    index_test = []
    for cl in classes:
        # Find index of class cl data point
        idx_cl = np.nonzero(target==cl)[0]
        idx_test_cl = idx_cl[0:n_test_per_class]
        idx_train_cl = idx_cl[n_test_per_class:]
        index_test.append(idx_test_cl)
        index_train.append(idx_train_cl)

    # take same num of samples from all classes
    ix_train = np.concatenate(index_train)
    ix_test = np.concatenate(index_test)
    
    X_train = data[ix_train]
    X_test = data[ix_test]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test