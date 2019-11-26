from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from fusion import *
        
netParams = snn.params('network.yaml')

device = torch.device('cuda')

net = Network(netParams, 'emg', 'dvsCropped').to(device)

net.load_state_dict(torch.load('Trained/fusionGesture.pt'))

testingSet = fusionDataset(
        samples     =np.loadtxt('test.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=netParams['simulation']['tSample'],
    )
testLoader = DataLoader(dataset=testingSet, batch_size=1, shuffle=True, num_workers=1)

stats = snn.utils.stats()

for trials in range(10):
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
            stats.print(
                    trials, i
                )
    stats.update()

print(
    f'Accuracy: {np.mean(np.array(stats.testing.accuracyLog)):.4g} \\pm {np.std(np.array(stats.testing.accuracyLog)):.4g}'
)

genLoihiParams(net)
plt.show()