import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from slayerLoihi import spikeLayer, quantizeWeights
from learningStats import learningStats
from dvsGesture import *
        
device = torch.device('cuda:3')
netParams = snn.params('network.yaml')
net = Network(netParams).to(device)
net.load_state_dict(torch.load('Trained/dvsGesture.pt'))

error = snn.loss(netParams, spikeLayer).to(device)

testingSet = IBMGestureDataset(datasetPath ='../data/', 
                               sampleFile  ='../test.txt',
                               samplingTime=netParams['simulation']['Ts'],
                               sampleLength=1450)

testLoader  = DataLoader(dataset=testingSet , batch_size=1, shuffle=True, num_workers=1)

stats = learningStats()

for i, (input, target, label) in enumerate(testLoader, 0):
    net.eval()
    with torch.no_grad():
        input  = input.to(device)
        target = target.to(device) 

    output = net.forward(input)

    stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
    stats.testing.numSamples     += len(label)
    # loss = error.numSpikes(output, target)
    # stats.testing.lossSum += loss.cpu().data.item()
    stats.print(0, i)

genLoihiParams(net)

plt.show()
