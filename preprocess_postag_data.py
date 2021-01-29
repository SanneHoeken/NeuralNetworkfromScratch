import numpy as np
from datasets import load_dataset
from itertools import chain
from gensim.models import KeyedVectors
import pickle

def preprocess_conll_data(dataset, embedding_model, dimensions):
    """
    Takes the CoNLL dataset, an embedding model and the dimensionality
    of its embeddings as input, extracts the tokens and target pos tags
    as lists and returns the tokens as word embeddings and the pos tags 
    """
    # make one list from the extracted list of lists of tokens in the dataset
    tokens = list(chain.from_iterable(dataset['tokens']))
    # make one list from extracted the list of lists of pos tags in the dataset
    pos_tags = list(chain.from_iterable(dataset['pos_tags']))

    # transform tokens to embeddings
    inputs_list = []
    # iterate over every token
    for token in tokens:
        # get embedding from pretrained model
        if token in embedding_model:
            vector = embedding_model[token]
        # get zero-filled vector if token not in model
        else:
            vector = [0]*dimensions
        inputs_list.append(vector)

    return inputs_list, pos_tags


def create_output_values(targets, output_nodes, activation_function):
    """
    Takes a list of target labels and transforms them in a list with 
    the length of the specified number of output nodes and transforms 
    the values into a range that corresponds to the given activation function.
    For the sigmoid function this is a value between 0 and 1 and 
    for a hyperbolic tangent function this is a value between -1 and 1. 
    The index corresponding to the target label is assigned the highest value.
    """
    targets_output_list = []
    # iterate over targets
    for target in targets:
        if activation_function == 'sigmoid':
            # create an array filled with 0.01's
            target_value = np.zeros(output_nodes) + 0.01
        elif activation_function == 'tanh':
            # create an array filled with -0.99's
            target_value = np.zeros(output_nodes) - 0.99
        # tag label is the index of the array that is set to 0.99
        target_value[target] = 0.99
        targets_output_list.append(target_value)

    return targets_output_list

def main():

    output_nodes = 12

    # PREPROCESS TRAINING DATA

    # load embedding model trained on COW corpus, SOURCE: https://github.com/clips/dutchembeddings
    embedding_model = KeyedVectors.load_word2vec_format("data/cow-embeddings-320/cow-big.txt", binary=False)
    # load the Dutch training CoNLL-2002 dataset
    train_dataset = load_dataset("conll2002", "nl", split='train')
    # get list of input tokens as embeddings and list of target labels
    inputs_list, targets_list = preprocess_conll_data(train_dataset, embedding_model, 320)
    # save the inputs list as txt-file
    pickle.dump(inputs_list, open('./data/conll2002/train_inputs.txt', 'wb'))
    
    # create output values for sigmoid as activation function
    targets_output_list = create_output_values(targets_list, output_nodes, 'sigmoid')
    # save the list with output values as txt-file
    pickle.dump(targets_output_list, open('./data/conll2002/train_targets_output.txt', 'wb'))
    
    # create output values for tanh as activation function
    targets_output_list = create_output_values(targets_list, output_nodes, 'tanh')
    # save the list with output values as txt-file
    pickle.dump(targets_output_list, open('./data/conll2002/train_targets_tanh_output.txt', 'wb'))
    
    # PREPROCESS VALIDATION DATA

    # load the Dutch validation CoNLL-2002 dataset
    validation_dataset = load_dataset("conll2002", "nl", split='validation')
    # get list of input tokens as embeddings and list of target labels
    inputs_list, targets_list = preprocess_conll_data(validation_dataset, embedding_model, 320)
    # save the inputs list and targets list as txt-files
    pickle.dump(inputs_list, open('./data/conll2002/validation_inputs.txt', 'wb'))
    pickle.dump(targets_list, open('./data/conll2002/validation_targets.txt', 'wb'))
    
    # PREPROCESS TEST DATA

    # load the Dutch test CoNLL-2002 dataset
    test_dataset = load_dataset("conll2002", "nl", split='test')
    # get list of input tokens as embeddings and list of target labels
    inputs_list, targets_list = preprocess_conll_data(test_dataset, embedding_model, 320)
    # save the inputs list and targets list as txt-files
    pickle.dump(inputs_list, open('./data/conll2002/test_inputs.txt', 'wb'))
    pickle.dump(targets_list, open('./data/conll2002/test_targets.txt', 'wb'))

if __name__ == "__main__":
    main()