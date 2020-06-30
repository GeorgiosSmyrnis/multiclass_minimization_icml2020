#!/bin/bash

set -e

mkdir -p './results_one_vs_all_fashion'
mkdir -p './models_one_vs_all_fashion'

for (( i = 0; i < 5; i++ )); do

  mkdir -p './models'

  python ./fashion_mnist_training.py ./models/fashion_mnist.h5

  mkdir -p './results'

  python ./multiclass_minim_one_vs_all_stable.py fashion_mnist.h5 [0.9,0.75,0.5,0.25,0.1] fashion_mnist

  mv ./results ./results_one_vs_all_fashion/results_extra_${i}

  mv ./models ./models_one_vs_all_fashion/models_${i}

done
