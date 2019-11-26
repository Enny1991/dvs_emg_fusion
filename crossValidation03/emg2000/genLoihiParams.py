from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from emgGesture import *

netParams = snn.params('network.yaml')

# Define the cuda device to run the code on.
device = torch.device('cuda:1')

# Create network instance.
net = Network(netParams).to(device)


# load saved net
net.load_state_dict(torch.load('Trained/emgGesture.pt'))
testingSet = handEmgDataset(samples     =np.loadtxt('../test.txt').astype(int),
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=200)
testLoader = DataLoader(dataset=testingSet, batch_size=4, shuffle=True, num_workers=1)

# generate Loihi parameters
stats = snn.utils.stats()

for trials in range(10):
	for i, (input, target, label) in enumerate(testLoader, 0):
		net.eval()
		with torch.no_grad():
			input  = input.to(device)
			target = target.to(device) 
			
			output, count = net.forward(input)

			stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
			stats.testing.numSamples     += len(label)

			# loss = error.numSpikes(output, target)
			# stats.testing.lossSum += loss.cpu().data.item()
			stats.print(
	                trials, i, 
	                header=['Spike Count:' + str(torch.sum(count, dim=0).tolist())],
	            )
	stats.update()

print(
	f'Accuracy: {np.mean(np.array(stats.testing.accuracyLog)):.4g} \\pm {np.std(np.array(stats.testing.accuracyLog)):.4g}'
)

genLoihiParams(net)
plt.show()
