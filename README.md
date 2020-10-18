Code for *Generalized Boosting* (Arun Sai Suggala, Bingbin Liu, Pradeep Ravikumar). To appear in NeurIPS 2020.

## File structure
* `main.py`: program entry for all methods except XGBoost.
  * `hp_tune.py`: program entry for hyperparameter tuning, i.e. splitting train set to training and validation.
* `xgb.py`: program entry for XGBoost.
* `models/`: model files. Including model definitions and training / testing loops.
* `data/`: code for data handling, e.g. data loaders.
* `datasets/`: folder for datasets. Currently empty; many datasets except (connect4, convex, mnist_rot) will be downloaded or generated as needed (see `data/data_loader.py` for details).
* `scripts/`: example training scripts. Please run the scripts in the main directory with `./scripts/{some_script}.sh` (rather than in the `script/` subfolder).

## Hyperparamters
Here are some sample commands to for some of the datasets

* CONNECT4:
`main.py --epochs=80 --lr=0.01 --momentum=0.9 --iterations 2 --width 1024 --weight-decay 0.001 --dataset=connect4 --basic-block=fc --algorithm=denseCompBoost --name=connect4_dense --scheduler-tolerance=4000`

* CONVEX:
`main.py --epochs=180 --lr=0.01 --momentum=0.9 --iterations 8 --width 128 --weight-decay 0.004 --dataset=convex --basic-block=conv_small --algorithm=denseCompBoost --name=convex_dense_small_conv --scheduler-tolerance=2000`

* COVTYPE:
`main.py --epochs=15 --lr=0.01 --momentum=0.9 --iterations=10 --width=4096 --weight-decay 0.0001 --dataset=covtype --basic-block=fc --algorithm=denseCompBoost  --scheduler-tolerance=10000`

* FASHIONMNIST:
`main.py --epochs=120 --lr=0.01 --momentum=0.9 --iterations 4 --width 128 --weight-decay 0.001 --dataset=fashionmnist --basic-block=conv_small --algorithm=denseCompBoost --scheduler-tolerance=7000`

* LETTER:
`main.py --epochs=150 --lr=0.01 --iterations 3 --width 1024 --weight-decay 0.002 --dataset=letter --basic-block=fc --algorithm=denseCompBoost --scheduler-tolerance=2800 --momentum=0.9 --seed=2`

* MNIST:
`main.py --epochs=80 --lr=0.01 --momentum=0.9 --iterations 2 --width 1024 --weight-decay 0.002 --dataset=mnist --basic-block=fc --algorithm=denseCompBoost --name=mnist_dense_hp --scheduler-tolerance=7000`

* SIM3:
`main.py --epochs=20 --lr=0.01 --momentum=0.9 --iterations 8 --width 1024 --weight-decay 0.0005 --dataset=sim3 --basic-block=fc --algorithm=denseCompBoost  --scheduler-tolerance=16000`

* SVHN:
`main.py --epochs=80 --lr=0.01 --momentum=0.9 --iterations 5 --width 128 --weight-decay 0.002 --dataset=svhn --basic-block=conv_small --algorithm=denseCompBoost --name=svhn_dense_small --scheduler-tolerance=8000`


