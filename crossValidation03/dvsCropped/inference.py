import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from dvsGesture import *

if __name__ == '__main__':
    # Define the cuda device to run the code on. 
    device = torch.device('cuda')

    # Read network configuration
    netParams = snn.params('network.yaml')
    net = Network(netParams).to(device)

    torch.nn.utils.remove_weight_norm(net.fc1, name='weight')
    torch.nn.utils.remove_weight_norm(net.fc2, name='weight')
    torch.nn.utils.remove_weight_norm(net.conv1, name='weight')
    torch.nn.utils.remove_weight_norm(net.conv2, name='weight')
    torch.nn.utils.remove_weight_norm(net.conv3, name='weight')

    fc1   = np.load('Trained/fc1Weights.npy').reshape(net.fc1.weight.shape)
    fc2   = np.load('Trained/fc2Weights.npy').reshape(net.fc2.weight.shape)
    
    conv1 = np.load('Trained/conv1Weights.npy').reshape(net.conv1.weight.shape)
    conv2 = np.load('Trained/conv2Weights.npy').reshape(net.conv2.weight.shape)
    conv3 = np.load('Trained/conv3Weights.npy').reshape(net.conv3.weight.shape)

    net.fc1  .weight.data = torch.FloatTensor(fc1).to(device)
    net.fc2  .weight.data = torch.FloatTensor(fc2).to(device)

    net.conv1.weight.data = torch.FloatTensor(conv1).to(device)
    net.conv2.weight.data = torch.FloatTensor(conv2).to(device)
    net.conv3.weight.data = torch.FloatTensor(conv3).to(device)

    # Create snn loss instance
    error = snn.loss(netParams, snn.loihi).to(device)

    testingSet = handDvsDataset(samples     =np.loadtxt('../test.txt').astype(int),
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=2000)
    testLoader = DataLoader(dataset=testingSet, batch_size=1, shuffle=False, num_workers=1)
    # testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=True, num_workers=4)

    # Learning stats instance.
    stats = snn.utils.stats()

    for i, (input, target, label) in enumerate(testLoader, 0):
        net.eval()
        
        with torch.no_grad():
            input  = input.to(device)
            target = target.to(device) 

            output, count = net.forward(input)

            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()

        stats.print(
            0, i, 
            header=['Spike Count:' + str(torch.sum(count, dim=0).tolist())],
        )
