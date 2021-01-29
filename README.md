# NeuralNetworkfromScratch

This project is part of the master course Machine Learning for NLP, which is part of the Humanities Research master at VU University Amsterdam.
January 2021.

### Project

In this project I built a simple one-layer neural network from scratch. I performed the NLP-task of tagging part of speech through this network and experimented with different parameters to improve the performance of the network as much as possible. 

## Getting started

### Requirements

This codebase is written entirely in Python 3.7. requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:

```
pip install -r requirements.txt
```

Or via conda:

```
conda install --file requirements.txt
```


## Using the main programs

To run the main programs below it is necessary to download the file 'cow-big.txt' from https://github.com/clips/dutchembeddings and put it in the folder named **cow-embeddings-320** which is a subfolder of the **data** folder. Also make sure that in this **data** folder there is an empty folder called **conll2002**.

1. **Preprocessing data.**

  This program can be run by calling:

  ```
  python preprocess_postag_data.py
  ```
  
  After the execution of the program, preprocessed versions of the necessary loaded datafiles will be stored in in **/data/conll2002**.

2. **Run the hyperparameter experiment.**

  This program can be run by calling:

  ```
  python postag_hyperparameter_experiment.py
  ```

  After the execution of the program, the performance of all possible combinations of selected value ranges for each of three hyper parameters is tested on the development set and the results are stored as **postag_hyperparameter_experiments.csv** in your current directory.

3. **Run the activation function experiment.**

  This program can be run by calling:

  ```
  python postag_activationfunction_experiment.py
  ```

  After the execution of the program, the performance of two activation functions is tested on the development set and the results are stored as **postag_activationfunction_experiments.csv** in your current directory.

4. **Get results on the test set.**

  This program can be run by calling:

  ```
  python postag_train_test.py
  ```
  
  After the execution of the program, the performance of the neural netwerk with the most optimal (according to the experiment results) parameter settings is tested on the test data. A classification report and confusion matrix are printed.
  

## Author
- Sanne Hoeken (student number: 2710599)