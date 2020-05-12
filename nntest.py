import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class ExampleNetwork(torch.nn.Module):
    def __init__(self, inputsize, hiddenSize, outputSize):
        super().__init__()
        self.layer1 = nn.Linear(inputsize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, outputSize)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

#   make a network
net = ExampleNetwork(2, 16, 1)

#   send the network to your device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  #   use this one if you dont have cuda
net.to(device)

#   make some data
inputs = torch.tensor([1.0, 2.0])
target = torch.tensor(5.0)

#   put the data on the device .__. cumbersome yes
inputs.to(device)
target.to(device)

#   pick a network stepping function and error function
#   #   Adam slowly reduces the learning rate each time step is called
optimizer = optim.Adam(net.parameters(), lr=0.001)  
#   #   a loss function... it's common, but technically you dont even need this. 
#   #   #   you could just subtract the output from the target data
lossFunction = torch.nn.MSELoss()   

#   train your network
numTrainingRounds = 1000
for i in range(numTrainingRounds):
    #   zero the derivatives !!! ALWAYS DO THIS BEFORE CALLING backward() !!!
    #   #   or else you will be using last training round's derivatives... its wrong!
    net.zero_grad() 

    output = net(inputs)    #   run data through neural network, to get its output
    loss = lossFunction(output, target) #   compute the error
    print(loss) #   the error number should decrease as it learns
                #   IF IT DOESNT... SOMETHING IS WRONG. AAAAA!!!

    loss.backward() #   compute derivatives throughout network (backpropogation)
    optimizer.step()    #   tweak network weights based on derivatives computed during backward()

