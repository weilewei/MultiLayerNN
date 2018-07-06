from phylanx.ast import *
import numpy as np


# To keep the randomness the same each time we run the code
# This can be removed if needed

# np.random.seed(1)


@Phylanx(debug=True)
def sigmoid(x):
    '''Keeps the output in the range of -1 to 1 with a smooth transition'''
    return 1 / (1 + np.exp(-x))


@Phylanx(debug=True)
def sigmoid_derivative(x):
    '''Derivative of the sigmoid function'''
    return x * (1 - x)


@Phylanx(debug=True)
def generate_network(network_shape, weight_arrays):
    '''
    Given a shape of the network, generate randomized weight matrices for the network
    '''
    cur_idx = 0
    next_idx = 0
    weight_array = 0
    local_weight_arrays = weight_arrays
    print("local_weight_arrays", local_weight_arrays)
    for i in range(0, len(network_shape) - 1):
        cur_idx = i
        next_idx = i + 1
        # Rows correspond to next set of nodes
        # Columns correspond to current set of nodes
        weight_array = 2 * np.random([network_shape[next_idx], network_shape[cur_idx]]) - 1
        local_weight_arrays += weight_array

    return local_weight_arrays


@Phylanx(debug=True)
def run_network(outputs, input, network_shape, network_weights):
    '''
    Given a trained network and the input(s), predict the possible output
    '''
    # Rows in the weight matrix correspond to nodes of the next layer
    # whereas columns correspond to nodes of the previous layer
    current_output = 0
    current_output_temp = 0
    local_input = input
    local_network_weights = network_weights
    for network_weight in local_network_weights:
        current_output_temp = np.dot(network_weight, local_input)
        # Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        outputs += current_output
        local_input = current_output

    return np.transpose(current_output)


@Phylanx(debug=True)
def train_network(input, output, predict_outputs, training_rate, network_shape, network_weights, deltas):
    '''
    Given an untrained network, inputs and expected outputs, train the network
    '''
    # print network_weights
    # print output
    # Our predicted outputs
    current_predict_outputs = predict_outputs
    current_output_temp = 0
    current_output = 0
    local_network_weights = network_weights
    local_input = input
    final_error = 0
    final_delta = 0
    next_error = 0
    next_delta = 0
    cur_delta = 0
    back_idx = 0
    cur_weight_idx = 0
    # print("local_network_weights", local_network_weights)
    for network_weight in local_network_weights:
        print("network_weight", network_weight)
        print("local_input", local_input)
        current_output_temp = np.dot(network_weight, local_input)
        # Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        current_predict_outputs += current_output
        local_input = current_output

    # # This will be in the reverse order
    # # Deltas will contain the error along with a few other terms which we come across
    # # due to how we formulate gradient descent of the neural network
    # # deltas = []
    # local_deltas = deltas
    # # We get these deltas according to the formula for gradient descent
    # final_error = output - current_predict_outputs[len(current_predict_outputs) - 1]
    # final_delta = final_error * sigmoid_derivative(current_predict_outputs[len(current_predict_outputs) - 1])
    # deltas += final_delta
    #
    # cur_delta = final_delta
    # back_idx = len(current_predict_outputs) - 2
    #
    # # Delta for layer i requires the weight matrix, delta of layer i+1 and expected output of layer i
    # # Going backwards (Backprop)
    # for network_weight in local_network_weights[::-1][:-1]:
    #     next_error = np.dot(np.transpose(network_weight), cur_delta)
    #     next_delta = next_error * sigmoid_derivative(current_predict_outputs[back_idx])
    #     deltas = + next_delta
    #     cur_delta = next_delta
    #     back_idx -= 1
    #
    # cur_weight_idx = len(local_network_weights) - 1
    #
    # # These deltas will be in the reverse order, so we move backwards through the layers
    # for delta in deltas:
    #     input_used = None
    #     if cur_weight_idx - 1 < 0:
    #         input_used = local_input
    #     else:
    #         input_used = current_predict_outputs[cur_weight_idx - 1]
    #
    #     # The weights of layer i are changed based on the input to layer i (or the output of layer i-1) and the delta of layer i
    #     # This is again due to the formulation of gradient descent
    #     local_network_weights[cur_weight_idx] += training_rate * np.dot(delta, np.transpose(input_used))
    #     cur_weight_idx -= 1
    #
    # # print local_network_weights
    # return local_network_weights


@Phylanx(debug=True)
def train_network_main(input, outputs, predict_outputs, training_rate, network_shape, network_weights, deltas):
    '''
    Take untrained weights and return trained weights for the neural network
    '''
    # Train the network multiple times to make it more accurate
    # local_input = input
    # print("train_network_main-local_input", local_input, '\n')
    local_outputs = outputs
    local_predict_outputs = predict_outputs
    local_network_shape = network_shape
    weight_arrays = network_weights
    for i in range(10000):
        weight_arrays = train_network(input, local_outputs, local_predict_outputs, training_rate,
                                      local_network_shape, weight_arrays, deltas)
        # print("weight_arrays", weight_arrays)
    return weight_arrays


inputs = np.transpose(np.array([[0, 0, 1, 1],
                                [1, 1, 1, 1],
                                [1, 0, 1, 1]]))

outputs = np.transpose(np.array([[0],
                                 [1],
                                 [1]]))

shape = [4, 3, 2, 1]

weight_arrays = []
weights = generate_network(shape, weight_arrays)
# print(weights)

training_rate = 0.3
# print("This is weights", weights)
predict_outputs = []
deltas = []
weights = train_network_main(inputs, outputs, predict_outputs, training_rate, shape, weights, deltas)

test_input = np.transpose(np.array([[0, 0, 1, 1]]))
predict_outputs = []
test_output = run_network(predict_outputs, test_input, shape, weights)
print(test_output)
