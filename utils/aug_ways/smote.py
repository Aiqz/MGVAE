import numpy as np

def smote(data, targets):
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    aug_data = []
    aug_label = []

    print("#------ Augmenting with SMOTE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))

        # Augmentation with SMOTE ( k-nearest )
        
        for i in range(class_len):

            for j in range(class_len):
                
                class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
        sorted_idx = np.argsort(class_dist)

        for i in range(n_max - class_len):
            # print(i)
            lam = np.random.uniform(0, 1)
            row_idx = i % class_len
            col_idx = int((i - row_idx) / class_len) % (class_len - 1)
            new_data = np.round(
                lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])
            # print(new_data.type)
            # aug_data.append(new_data.dbyte())
            aug_data.append(new_data.numpy())
            aug_label.append(k)
    # return aug_data, aug_label
    print("#------ Augment Done! ------#")
    print(len(aug_data))
    return np.array(aug_data), np.array(aug_label)