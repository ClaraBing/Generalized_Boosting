#!/bin/bash

dataset='sim3'
basic_block='fc'

width_incre=2
tolerance=8000

min_lr=1e-6
mm=0.9
epochs=120

optimizer='SGD'

# Choose an algorithm
alg='denseCompBoost'

# Can also choose from:
# 1. Complex greedy
# alg='cmplxCompBoost'
# 2. Standard greedy
# alg='stdCompBoost'
# 3. Joint (end-to-end)
# alg='joint'
# 4. AdaBoost
# alg='adaBoost'
# 5. Additive Feature Boost
# alg='featureBoost'

subsample=0
transform='none'

for run in 1, 2
do
  for wd in 1e-5 4e-5 1e-4 4e-4 1e-3 4e-3
  do
    for lr in 0.01
    do
      for width in 1024
      do
      for iterations in 10
      do
        suffix='hpTune'$hp_tune'_d'$dataset'_'$alg'_block'$basic_block'_depth'$iterations'_lr'$lr'_w'$width'_wd'$wd'_run'$run'_T'$transform'_E'$epochs'_P'$tolerance
        CUDA_VISIBLE_DEVICES=0 python -m hp_tune.py \
          --dataset=$dataset \
          --iterations=$iterations \
          --basic-block=$basic_block \
          --algorithm=$alg \
          --scheduler-tolerance=$tolerance \
          --suffix=$suffix \
          --width=$width \
          --optimizer=$optimizer \
          --lr=$lr \
          --min-lr=$min_lr \
          --weight-decay=$wd \
          --momentum=$mm \
          --epochs=$epochs \
          --transform=$transform \
          --subsample=$subsample
      done
      done
    done
  done
done

