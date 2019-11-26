import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from emgGesture import *

if __name__ == '__main__':
    # Define the cuda device to run the code on. 
    device = torch.device('cuda')

    # Read network configuration
    netParams = snn.params('network.yaml')
    net = Network(netParams).to(device)

    torch.nn.utils.remove_weight_norm(net.fc1, name='weight')
    torch.nn.utils.remove_weight_norm(net.fc2, name='weight')
    torch.nn.utils.remove_weight_norm(net.fc3, name='weight')

    fc1   = np.load('Trained/fc1Weights.npy').reshape(net.fc1.weight.shape)
    fc2   = np.load('Trained/fc2Weights.npy').reshape(net.fc2.weight.shape)
    fc3   = np.load('Trained/fc3Weights.npy').reshape(net.fc3.weight.shape)

    delay1 = np.load('Trained/delay1.npy').reshape(net.delay1.delay.shape)
    delay2 = np.load('Trained/delay2.npy').reshape(net.delay2.delay.shape)

    # fc1   = np.loadtxt('Trained/fc1Weights.txt').reshape(net.fc1.weight.shape)
    # fc2   = np.loadtxt('Trained/fc2Weights.txt').reshape(net.fc2.weight.shape)
    # fc3   = np.loadtxt('Trained/fc3Weights.txt').reshape(net.fc3.weight.shape)

    # delay1 = np.loadtxt('Trained/delay1.txt').reshape(net.delay1.delay.shape)
    # delay2 = np.loadtxt('Trained/delay2.txt').reshape(net.delay2.delay.shape)

    net.fc1  .weight.data = torch.FloatTensor(fc1).to(device)
    net.fc2  .weight.data = torch.FloatTensor(fc2).to(device)
    net.fc3  .weight.data = torch.FloatTensor(fc3).to(device)

    net.delay1.delay.data = torch.FloatTensor(delay1).to(device)
    net.delay2.delay.data = torch.FloatTensor(delay2).to(device)

    # net.delay1.delay.data = torch.FloatTensor(delay1).to(device) * 0
    # net.delay2.delay.data = torch.FloatTensor(delay2).to(device) * 0

    # Create snn loss instance
    error = snn.loss(netParams, snn.loihi).to(device)

    testingSet = handEmgDataset(samples     =np.loadtxt('../test.txt').astype(int),
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

        # print(torch.sum(input))
        # print(torch.sum(output, dim=4).flatten())

        # aer = np.argwhere(output.squeeze().cpu().data.numpy())

        # plt.figure(figsize = (18, 5))
        # plt.plot(aer[:,1], aer[:,0], '.')
        # plt.show()

        # break