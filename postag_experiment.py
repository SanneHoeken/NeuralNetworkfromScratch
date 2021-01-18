import numpy as np
import pickle, csv
from sklearn.metrics import accuracy_score
from neural_network import neuralNetwork

def main():

    # get training data
    train_inputs_list = pickle.load(open('./data/conll2002/train_inputs.txt', 'rb'))
    train_targets_output_list = pickle.load(open('./data/conll2002/train_targets_output.txt', 'rb'))

    # get validation data
    val_inputs_list = pickle.load(open('./data/conll2002/validation_inputs.txt', 'rb'))
    val_targets_list = pickle.load(open('./data/conll2002/validation_targets.txt', 'rb'))

    # Random initialize network 5 times for all different hyperparameter combinations
    # Train each initialized network on the training data
    # Get the results of the validation data and save the accuracy scores to a csv

    input_nodes = 320
    output_nodes = 12

    learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epochs_list = [1, 2, 3, 4, 5, 6, 7, 10]
    hidden_nodes_list = [100, 150, 200, 250, 300, 350]

    results = []

    for i in range(5):
        for epochs in epochs_list:
            for learning_rate in learning_rate_list:
                for hidden_nodes in hidden_nodes_list:
                    # create instance of neural network
                    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

                    # train the neural network
                    n.train(train_inputs_list, train_targets_output_list, epochs)

                    # query predictions for the validation data
                    predictions = []
                    for inp in val_inputs_list:
                        # query the network
                        output = n.forward(np.array(inp, ndmin=2).T)[0]
                        # label is index with highest value 
                        predictions.append(np.argmax(output))

                    # add parameter settings with accuracy score to results
                    accuracy = accuracy_score(val_targets_list, predictions)
                    result = {'epochs': epochs, 'learning_rate': learning_rate, 'hidden_nodes': hidden_nodes, 'accuracy': accuracy}
                    results.append(result)
                    print(result)

    # save results as csv-file
    with open("postag_hyperparameter_experiments.csv", 'w', newline='') as outfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    

if __name__ == "__main__":
    main()