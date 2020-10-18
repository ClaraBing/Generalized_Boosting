def get_num_classes(dataset):
    if dataset in ['svhn', 'cifar10', 'fashionmnist', 'mnist', 'mnist_rot']:
        return 10
    elif dataset == 'letter':
        return 26
    elif dataset == 'covtype':
        return 7
    elif dataset == 'connect4':
        return 3
    elif dataset in ['sim1', 'sim2', 'sim3', 'convex']:
        return 2

