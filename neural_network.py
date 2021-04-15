# Library imports
import numpy as np
from scipy.stats import norm

# Load datasets
train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")

# Sanity checks
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


def generate_gaussian_weights(num_neurons, num_features, random_state = None):
    '''
    Generate weights taken from the Gaussian distribution, and based on the number of neurons within the hidden layer
    as well as the number of features of the dataset
    
    Output is weights matrix of size (num_neurons, num_features)
    '''
    
    weights = norm.rvs(size = [num_neurons, num_features], random_state=random_state)
    
    return weights

def generate_gaussian_bias(num_neurons, random_state = None):
    '''
    Generate bias vector from the Gaussian distribution, and based on the number of neurons within the hidden layer
    
    Output is bias vector of size (num_neurons)
    '''
    
    bias = norm.rvs(size = num_neurons, random_state=random_state)
    
    return bias


def calc_z(data_vector, weights_matrix, bias_vector):
    '''
    Calculate the z value for all the neurons within the specific hidden layer, obtained by taking the dot product
    between the weights matrix and data vector.  The bias vector is then added onto the product of the two.
    
    The output vector then represents the input value to be used for the activation function of all the neurons within
    the specific hidden layer.
    '''
    
    return weights_matrix.dot(data_vector) + bias_vector


def run_activation_func(activation_func, z):
    '''
    Calculates the value after the z has been computed and puts it inside the non-linear activation function
    that we have for that hidden layer
    '''
    
    if activation_func == 'relu':
        return np.maximum(0, z)
    
    if activation_func == 'softmax':
        z = z - np.max(z) # we're adding this such that it doesn't overflow with very large nums
        return np.divide(np.exp(z), np.sum(np.exp(z)))


def encode_label_vector(label_vector):
    '''
    Encode the label vector of our dataset, such that we can use it for the computation of the MSE.
    This is because the labels are labelled as integers 0 to 9, whereas for MSE to work, we need to
    create a array vector of size 10 for every single observation, where the value 1 is set on the index of the correct observation
    '''
    
    num_classes = np.unique(train_label).size
    
    encoded_label_vector = []
    
    for label in label_vector:
        encoded_label = np.zeros(num_classes)
        encoded_label[label] = 1
        encoded_label_vector.append(encoded_label)
    
    encoded_label_vector = np.array(encoded_label_vector)
    
    return encoded_label_vector
    

def calculate_MSE(pred, actual):
    '''
    Calculates the Mean Squared Error between the prediction of the NN and actual class
    
    Note:
    This works because the pred value is the softmax of the output of the NN and
    the actual is adjusted such that the values are between 0 and 1
    '''
    error = np.subtract(pred, actual)
    squared_error = np.square(error)
    return np.sum(squared_error)


def calc_delta_softmax(layer_output, encoded_label_vector):
    '''
    Calculates the delta value for the final/output layer, which will be used for backpropagation
    
    Note:
    This function expects to receive the output of the softmax activation function within the output layer
    and also the encoded label vector which would have values 0 and 1 exclusively
    
    delta = activation_output - y
    
    Output: vector of size num of classes to be predicted
    '''
    
    return np.subtract(layer_output, encoded_label_vector)


def calc_delta_hidden(l_plus_one_delta, l_plus_one_weights, layer_output):
    '''
    Calculates the delta value for the hidden layers, which will be used for backpropagation
    
    This function is only to be used to calculate the delta values of the hidden layers.  Output layer delta should be
    the calc_delta_softmax, and there is no delta term needed to be calculated for input layer (NN layer 1)
    
    delta = (weights for next layer . delta for next layer) .* (activation_func output for this layer .* (1 - activation_func output for this layer))
    '''

    first_part = (l_plus_one_weights.T).dot(l_plus_one_delta)
    second_part = np.multiply(layer_output, np.ones(layer_output.size) - layer_output)
    return  np.multiply(first_part, second_part)


# Neural Network Setup

# Static variables
HIDDEN_LAYERS: int = 2
HIDDEN_LAYERS_ACTIVATION_FUNC: list = ['relu', 'relu']
NUM_NEURONS: list = [5,7] # this should be a list containing int per hidden layer

if not (HIDDEN_LAYERS == len(HIDDEN_LAYERS_ACTIVATION_FUNC) == len(NUM_NEURONS)):
    raise Exception("The num of layers must have the appropriate num of activation funcs and num of neurons specified")
    
BATCH_SIZE: int = 1000
LEARNING_RATE: float = 0.001
NUM_EPOCHS: int = 2000

    
# Initialisation
encoded_label_vector = encode_label_vector(train_label)
num_classes = np.unique(train_label).size

weight_matrix = []
bias_vector = []
epoch_loss = []

# initialise the weights
for layer_num, layer in enumerate(range(HIDDEN_LAYERS)):
    
    # If we are instantiating the details for the first hidden layer, then make the following adjustments
    # which would otherwise be not required for subsequent hidden layers
    if layer_num == 0:
        # The input features would be the shape of our dataset instead of num of features from previous layer
        num_input_features = train_data.shape[1]
    else:
        num_input_features = NUM_NEURONS[layer_num - 1]
    
    # check how many neurons should be in this layer
    neuron_num = NUM_NEURONS[layer_num]
    
    layer_weights = generate_gaussian_weights(neuron_num, num_input_features)
    layer_bias = generate_gaussian_bias(neuron_num)
    
    weight_matrix.append(layer_weights)
    bias_vector.append(layer_bias)
    
    print(f'Weight and bias generated for hidden layer {layer_num + 1} with weight shape {weight_matrix[layer_num].shape} and bias shape of {bias_vector[layer_num].shape}')

   
# insantiate the parts for the output layer
# need to be very careful with the use of -1 indices, in the event that we incorporate output layer to our hidden layer variables
weight_matrix.append(generate_gaussian_weights(num_classes, NUM_NEURONS[-1]))
bias_vector.append(generate_gaussian_bias(num_classes))

weight_matrix = np.array(weight_matrix, dtype=object)


### Getting the network running

observation_idx_current: int = 0 # basically keeps a tally of where we are up to in the dataset - used for batches
ready_for_exit: bool = False

for epoch_num in range(NUM_EPOCHS):
    batch_loss = []

    if (observation_idx_current + BATCH_SIZE) > train_data.shape[0]:
        BATCH_SIZE = train_data.shape[0] - observation_idx_current
        ready_for_exit = True # because there is no more observations for next epoch

    # feedforward part
    # for the two variables below, the expected final state is numpy_array(representing each layer)
    # in a list (representing each observation)
    # in a list (the final container of the object)
    layer_z = [[] for i in range(BATCH_SIZE)]
    layer_output = [[] for i in range(BATCH_SIZE)]

    for observation_idx, observation_val in enumerate(range(BATCH_SIZE)):

        for layer_num, layer in enumerate(range(HIDDEN_LAYERS)):

            if layer_num == 0:
                input_data = train_data[observation_idx_current + observation_idx]
            else:
                # extract the output of the previous layer
                input_data = layer_output[observation_idx][layer_num - 1]

            z = calc_z(input_data, weight_matrix[layer_num], bias_vector[layer_num])
            a = run_activation_func(HIDDEN_LAYERS_ACTIVATION_FUNC[layer_num], z)

            layer_z[observation_idx].append(z)
            layer_output[observation_idx].append(a)

        # Calculation for the output layer
        z = calc_z(layer_output[observation_idx][-1], weight_matrix[-1], bias_vector[-1])
        a = run_activation_func('softmax', z)
        layer_z[observation_idx].append(z) # to be used for backpropagation
        layer_output[observation_idx].append(a)

        # Calculate the error
        loss = calculate_MSE(layer_output[observation_idx][-1], encoded_label_vector[observation_idx_current + observation_idx]) 
        batch_loss.append(loss)

    epoch_loss.append(np.average(batch_loss))

    # Perform the backpropagation
    # instantiate the delta list obj for HIDDEN_LAYERS + output layer
    delta = [[0 for i in range(HIDDEN_LAYERS + 1)] for i in range(BATCH_SIZE)]
    d = [[0 for i in range(HIDDEN_LAYERS + 1)] for i in range(BATCH_SIZE)]

    for observation_idx, observation_val in enumerate(range(BATCH_SIZE)):
        # for the output layer
        delta[observation_idx][HIDDEN_LAYERS] = calc_delta_softmax(layer_output[observation_idx][-1], encoded_label_vector[observation_idx])
        d[observation_idx][HIDDEN_LAYERS] = np.outer(delta[observation_idx][HIDDEN_LAYERS], layer_output[observation_idx][HIDDEN_LAYERS - 1].T)

        for layer_num, layer in reversed(list(enumerate(range(HIDDEN_LAYERS)))):

            delta[observation_idx][layer_num] = calc_delta_hidden(delta[observation_idx][layer_num + 1], weight_matrix[layer_num + 1], layer_output[observation_idx][layer_num])
            #print(f'At observation {observation_idx} and layer num {layer_num}, the delta is {delta[observation_idx][layer_num]}')

            if layer_num == 0:
                # Because we're at layer 0 (the first hidden layer), then we need to use the raw data inputs as the layer_output of the input layer
                # note that the input_layer is not included in our layer_output list
                d[observation_idx][layer_num] = np.outer(delta[observation_idx][layer_num], train_data[observation_idx].T)
            else:
                d[observation_idx][layer_num] = np.outer(delta[observation_idx][layer_num], layer_output[observation_idx][layer_num - 1].T)


    # Time to calculate the average delta from across different observations
    d = np.array(d, dtype=object) # quite important to convert this list into a pure numpy array for avg next
    avg_d = np.average(d, axis = 0) # each array within the array, then represents the neurons within a layer


    # Time to update our weights, using the averaged delta from previous calc
    update = np.array(LEARNING_RATE * avg_d)
    weight_matrix = np.subtract(weight_matrix, update)

    observation_idx_current += BATCH_SIZE

    if ready_for_exit:
        print("Exiting early due to no more observations to feed in")
        break

print(epoch_loss)