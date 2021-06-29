# Fusion Learning
Fusion Learning [1] is a Federated Learning framework, which allows training a neural network in a distributed and privacy-preserving manner.
This repository contains an experiment on the accuracy of models trained with Fusion Learning.

# Experiment
In `main.py` a simulation of Fusion Learning is provided.
This simulates training a multi-layer perceptron on a number of different datasets.

# How to reproduce
To reproduce this experiment you will need to install the following:
1. Python 3.8
2. conda
3. pip

To install all the dependencies run:
```
conda env create --file environment.yml --name fusion-learning-env
```

Then, to run the experiment, execute:
```
conda activate fusion-learning-env
python main.py
```

The output of the experiment is the accuracy of the trained model.
