import numpy as np
import pickle, csv, time
from sklearn.metrics import accuracy_score
from neural_network import neuralNetwork

def main():

    # get training data
    train_inputs_list = pickle.load(open('./data/conll2002/train_inputs.txt', 'rb'))
    train_targets_output_list = pickle.load(open('./data/conll2002/train_targets_output.txt', 'rb'))

    # get validation data
    val_inputs_list = pickle.load(open('./data/conll2002/validation_inputs.txt', 'rb'))
    val_targets_list = pickle.load(open('./data/conll2002/validation_targets.txt', 'rb'))

    # set number of input and output nodes of the network
    input_nodes = 320
    output_nodes = 12

    # initialize lists with value ranges for the hyperparameters
    learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epochs_list = [1, 2, 3, 4, 5, 6, 7]
    hidden_nodes_list = [50, 100, 150, 200, 250, 300, 350]

    results = []

    # iterate over all different hyperparameter combinations
    for epochs in epochs_list:
        for learning_rate in learning_rate_list:
            for hidden_nodes in hidden_nodes_list:

                # store the parameter settings as a result dictionary
                result = {'epochs': epochs, 'learning_rate': learning_rate, 'hidden_nodes': hidden_nodes}
                
                # create instance of neural network
                n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate, 'sigmoid')

                # train the neural network and measure the time it takes
                start = time.perf_counter()
                n.train(train_inputs_list, train_targets_output_list, epochs)
                stop = time.perf_counter()

                # add training time to result
                result['time'] = stop - start

                # query predictions for the validation data
                predictions = []
                for inp in val_inputs_list:
                    # query the network
                    output = n.forward(np.array(inp, ndmin=2).T)[0]
                    # label is index with highest value 
                    predictions.append(np.argmax(output))

                # add accuracy score to result and append to list of results
                result['accuracy'] = accuracy_score(val_targets_list, predictions)
                results.append(result)

    # save results as csv-file
    with open("postag_hyperparameter_experiments.csv", 'w', newline='') as outfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    

if __name__ == "__main__":
    main()