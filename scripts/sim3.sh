#!/bin/bash

hp_tune=0

dataset='sim3'
basic_block='fc'

width_incre=2
tolerance=8000

lr=0.01
min_lr=1e-6
mm=0.9
epochs=250

optimizer='SGD'

# Choose an algorithm
alg='denseCompBoost'

wd=1e-6
width=128
iterations=2

subsample=0
transform='none'

# do multiple runs and take the average.
for run in 1 2
do
  suffix='hpTune'$hp_tune'_d'$dataset'_'$alg'_block'$basic_block'_depth'$iterations'_lr'$lr'_w'$width'_wd'$wd'_run'$run'_T'$transform'_E'$epochs'_P'$tolerance
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --hp-tune=$hp_tune \
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

