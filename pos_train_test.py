import numpy as np
import pickle
from sklearn.metrics import classification_report
from neural_network import neuralNetwork

def main():

    # get training data
    train_inputs_list = pickle.load(open('./data/conll2002/train_inputs.txt', 'rb'))
    train_targets_output_list = pickle.load(open('./data/conll2002/train_targets_output.txt', 'rb'))

    # get test data
    test_inputs_list = pickle.load(open('./data/conll2002/test_inputs.txt', 'rb'))
    test_targets_list = pickle.load(open('./data/conll2002/test_targets.txt', 'rb'))

    input_nodes = 320
    output_nodes = 12
    hidden_nodes = 100
    learning_rate = 0.3
    epochs = 3
    
    # create instance of neural network
    n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)
    
    # train the neural network
    n.train(train_inputs_list, train_targets_output_list, epochs)
    
    # query predictions for the validation data
    predictions = []
    for inp in test_inputs_list:
        # query the network
        output = n.forward(np.array(inp, ndmin=2).T)[0]
        # label is index with highest value 
        predictions.append(np.argmax(output))

    # print results
    print(classification_report(test_targets_list, predictions))


if __name__ == "__main__":
    main()