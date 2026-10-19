#INTI COMMIT :)

import torch


class Neuron:
    def __init__ (self, weights, bias=0.0):
        self.weights = torch.tensor(weights, dtype=torch.float32)

        self.bias = bias

    def forward(self, inputs):
        #give each connection a weight
        weighted_inputs = inputs * self.weights

        #combine incoming input
        total = weighted_inputs.sum() + self.bias
        print(type(total))
        # Activation keep positive values turn negative -> 0
        return torch.relu(total)

    
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        # each neron procces same input independely
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return torch.stack(outputs)
    
#Hidden layers

hidden_layer = Layer([
    Neuron(weights=[ 0.5, 0.2], bias=0.1),
    Neuron(weights=[-0.3, 0.8], bias=0.0),
    Neuron(weights=[ 0.7, -0.5], bias=0.2),
])

#Output

output_layer = Layer([
    Neuron(weights= [ 0.4, 0.6, -0.2], bias=-0.1),
])

inputs = torch.tensor([1.0, 2.0])
hidden = hidden_layer.forward(inputs)
output = output_layer.forward(hidden)

print("Hidden: ", hidden)
print("Output: ", output)

