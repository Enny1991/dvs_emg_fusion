from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from fusion import *

if __name__ == '__main__':
    # Define the cuda device to run the code on. 
    device = torch.device('cuda')

    # Read network configuration
    netParams = snn.params('network.yaml')
    net = Network(netParams, 'emg', 'dvsCropped').to(device)

    torch.nn.utils.remove_weight_norm(net.fc1, name='weight')

    fc1   = np.load('Trained/fc1Weights.npy').reshape(net.fc1.weight.shape)
    
    net.fc1  .weight.data = torch.FloatTensor(fc1).to(device)

    # Create snn loss instance
    error = snn.loss(netParams, snn.loihi).to(device)

    testingSet = fusionDataset(
        samples     =np.loadtxt('test.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=2000,
    )
    testLoader = DataLoader(dataset=testingSet, batch_size=1, shuffle=False, num_workers=1)
    # testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=True, num_workers=4)

    # Learning stats instance.
    stats = snn.utils.stats()

    for i, (emgInput, dvsInput, target, label) in enumerate(testLoader, 0):
        net.eval()
        with torch.no_grad():
            emgInput  = emgInput.to(device)
            dvsInput  = dvsInput.to(device)
            target    = target.to(device) 

            output = net.forward(emgInput, dvsInput)

            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            # loss = error.numSpikes(output, target)
            # stats.testing.lossSum += loss.cpu().data.item()
            stats.print(0, i)
