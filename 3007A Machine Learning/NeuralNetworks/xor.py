import numpy as np 
#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def f(x):
    return (x**2)*math.sin(2*math.pi*x) + 0.7

def point(xVals, yVals):
    x = xVals[0] + xVals[1]*np.random.randn()
    y = yVals[0] + yVals[1]*np.random.randn()
    arr = np.array([x, y])
    return arr

def dataGen(n, xVar, yVar):
    #This function returns 3 arrays: data, data0, data1
    #len(data) = len(data0) + len(data1) = n
    #Data contains n points generated by points()
    data = np.zeros((n, 3))
    data0 = np.array([0, 0]).reshape(1, 2)
    i0 = 0
    data1 = np.array([0, 0]).reshape(1, 2)
    i1 = 0
    for i in range(n):
        p = point(xVar, yVar).reshape(1, 2)
        if f(p[0,0]) > p[0,1]:
            data0 = np.append(data0, p, 0)
            p = np.append(p, 0).reshape(1, 3)
        else:
            data1 = np.append(data1, p, 0)
            p = np.append(p, 1).reshape(1, 3)
        data[i] = p
    data0 = np.delete(data0, 0, 0)
    data1 = np.delete(data1, 0, 0)
    return data, data0, data1

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

print("Initial hidden weights: ",end='')
print(*hidden_weights)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)


#Training algorithm
for _ in range(epochs):
	#Forward Propagation
	hidden_layer_activation = np.dot(inputs,hidden_weights)
	hidden_layer_activation += hidden_bias
	hidden_layer_output = sigmoid(hidden_layer_activation)

	output_layer_activation = np.dot(hidden_layer_output,output_weights)
	output_layer_activation += output_bias
	predicted_output = sigmoid(output_layer_activation)

	#Backpropagation
	error = expected_output - predicted_output
	d_predicted_output = error * sigmoid_derivative(predicted_output)
	print(d_predicted_output)
	
	error_hidden_layer = d_predicted_output.dot(output_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
	print(d_hidden_layer)

	#Updating Weights and Biases
	output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
	output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
	hidden_weights += inputs.T.dot(d_hidden_layer) * lr
	hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

print("Final hidden weights: ",end='')
print(*hidden_weights)
print("Final hidden bias: ",end='')
print(*hidden_bias)
print("Final output weights: ",end='')
print(*output_weights)
print("Final output bias: ",end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)

print("============================================================")

finalIn = np.array([0, 0])

hidden_layer_activation = np.dot(finalIn,hidden_weights)
hidden_layer_activation += hidden_bias
hidden_layer_output = sigmoid(hidden_layer_activation)

print(hidden_layer_output)

output_layer_activation = np.dot(hidden_layer_output,output_weights)
output_layer_activation += output_bias
predicted_output = sigmoid(output_layer_activation)

print(predicted_output)