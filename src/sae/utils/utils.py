def train_val_test_split(ids, train_frac=0.9, val_frac=0.05, seed=42):
    import random
    random.seed(seed)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test
