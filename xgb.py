from __future__ import print_function
import argparse, yaml
import xgboost as xgb
import numpy as np
from sklearn.metrics import auc, accuracy_score
from sklearn.metrics import make_scorer
import torch

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data.data_loader import get_torch_dataset
from data.data_utils import get_num_classes

# Training settings
parser = argparse.ArgumentParser(description='XGBOOST')
parser.add_argument('--n-estimators', nargs='+', type=int, default=[100], metavar='N',
                    help='number of estimators to fit (default: 100)')
parser.add_argument('--lr', nargs='+', type=float, default=[0.1], metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--max-depth', nargs='+', type=int, default=[10], metavar='MaxD',
                    help='max depth of each tree (default: 5)')
parser.add_argument('--colsample', nargs='+', type=float, default=[1.0], metavar='CS',
                    help='fractions of features to randomly sample to fit each tree (default: 1.0)')
parser.add_argument('--subsample', nargs='+', type=float, default=[1.0], metavar='SS',
                    help='fractions of data points to randomly sample to fit each tree (default: 1.0)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='sim3', metavar='DATASET',
                    choices=['cifar10', 'svhn', 'fashionmnist', 'convex', 'mnist_rot', # image datasets
                             'mnist', 'letter', 'covtype',  'connect4', # tabular datasets
                             'sim1', 'sim2', 'sim3'], # simulated datasets
                    help='Dataset to use. (default: "sim3")')
parser.add_argument('--n-threads', type=int, default=4, metavar='NT',
                    help='number of threads (default: 4)')


args = parser.parse_args()
np.random.seed(args.seed)

train_dataset, test_dataset = get_torch_dataset(dataset=args.dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


train_X = None
train_y = None
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.numpy(), target.numpy()
    if batch_idx == 0:
        train_X, train_y = data.reshape(data.shape[0], -1), target
        continue
    train_X = np.concatenate((train_X, data.reshape(data.shape[0], -1)), axis = 0)
    train_y = np.concatenate((train_y, target), axis = 0)

test_X = None
test_y = None
for batch_idx, (data, target) in enumerate(test_loader):
    data, target = data.numpy(), target.numpy()
    if batch_idx == 0:
        test_X, test_y = data.reshape(data.shape[0], -1), target
        continue
    test_X = np.concatenate((test_X, data.reshape(data.shape[0], -1)), axis = 0)
    test_y = np.concatenate((test_y, target), axis = 0)


# first do grid search to pick the best hyper-parameters
# A parameter grid for XGBoost

num_classes = get_num_classes(args.dataset)
scorer = None
objective = None
if num_classes == 2:
    scorer = make_scorer(roc_auc_score)
    objective = 'binary:logistic'
elif num_classes > 2:
    scorer = make_scorer(accuracy_score) # 'neg_log_loss'
    objective = 'multi:softmax'

params = {
        'learning_rate': args.lr,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample,
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators
        }

gbm = xgb.XGBClassifier(nthread = 1, objective= objective)
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
grid_search = GridSearchCV(gbm, params, scoring=scorer,
                          n_jobs=args.n_threads, cv=skf.split(train_X,train_y), verbose=3)
grid_search.fit(train_X, train_y)

print('\n All results:')
print(grid_search.cv_results_)
print('\n Best estimator:')
print(grid_search.best_estimator_)
print('\n Best hyperparameters:')
print(grid_search.best_params_)


# train using best hyper-parameters
if num_classes == 2:
    gbm = xgb.XGBClassifier(nthread = args.n_threads, max_depth=grid_search.best_params_['max_depth'], n_estimators=grid_search.best_params_['n_estimators'],
                        learning_rate=grid_search.best_params_['learning_rate'], objective= objective,
                        colsample_bytree=grid_search.best_params_['colsample_bytree'], subsample = grid_search.best_params_['subsample']).fit(train_X, train_y)
else:
    gbm = xgb.XGBClassifier(nthread = args.n_threads, max_depth=grid_search.best_params_['max_depth'], n_estimators=grid_search.best_params_['n_estimators'],
                        learning_rate=grid_search.best_params_['learning_rate'], objective= objective,  num_class=num_classes,
                        colsample_bytree=grid_search.best_params_['colsample_bytree'], subsample = grid_search.best_params_['subsample']).fit(train_X, train_y)

# testing
predictions = gbm.predict(test_X)
print('Test accuracy: {}'.format(accuracy_score(test_y, predictions)))

predictions = gbm.predict(train_X)
print('Train accuracy: {}'.format(accuracy_score(train_y, predictions)))

