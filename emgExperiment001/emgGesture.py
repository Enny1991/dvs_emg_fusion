from datetime import datetime
import pickle as pkl
import numpy as np
# import matplotlib
# if __name__ == '__main__': 
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import slayerSNN.optimizer as optim
from slayerSNN.learningStats import learningStats

actions = ['pinky', 'elle', 'yo', 'index', 'thumb']
label = {'pinky' : 0, 'elle' : 1, 'yo' : 2, 'index' : 3, 'thumb' : 4}

# Dataset definition
class handEmgDataset(Dataset):
    def __init__(self, samples, samplingTime, sampleLength):
        self.data         = pkl.load(open('../data/relax_spikes_2seconds.pkl', 'rb'))
        self.samples      = samples
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        # load event and label
        sampleID = self.samples[index]
        label = int(self.data['y'][sampleID])
        event = self.data['emg'][sampleID]

        inputSpikes = snn.io.event(
                        event[0], None, event[2], event[1]*1000
                        ).toSpikeTensor(torch.zeros((2,1,16,self.nTimeBins)),
                        samplingTime=self.samplingTime, 
                        randomShift=True)
        
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((5, 1, 1, 1))
        desiredClass[label,...] = 1
        
        return inputSpikes, desiredClass, label
        
    def __len__(self):
        return self.samples.shape[0]
        
# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.fc1   = torch.nn.utils.weight_norm(slayer.dense( 32, 128), name='weight')
        self.fc2   = torch.nn.utils.weight_norm(slayer.dense(128, 128), name='weight')
        self.fc3   = torch.nn.utils.weight_norm(slayer.dense(128,   5), name='weight')
        self.delay1= slayer.delay(128)
        self.delay2= slayer.delay(128)
        
    def forward(self, spike):
        spikeCount = []

        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1])) # 32
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 128
        spike = self.slayer.delayShift(spike, 1)

        spikeCount.append(torch.sum(spike).item())
        
        spike = self.delay1(spike)
        spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 128
        spike = self.slayer.delayShift(spike, 1)

        spikeCount.append(torch.sum(spike).item())
        
        spike = self.delay2(spike)
        spike = self.slayer.spikeLoihi(self.fc3  (spike)) # 128
        spike = self.slayer.delayShift(spike, 1)

        spikeCount.append(torch.sum(spike).item())

        return spike, torch.FloatTensor(spikeCount).reshape((1, -1)).to(spike.get_device())

    def clamp(self):
        self.delay1.delay.data.clamp_(0, 62)
        self.delay2.delay.data.clamp_(0, 62)

    def gradFlow(self, path):
        gradNorm = lambda x: torch.norm(x).item()/torch.numel(x)

        grad = []
        grad.append(gradNorm(self.fc1  .weight_g.grad))
        grad.append(gradNorm(self.fc2  .weight_g.grad))
        grad.append(gradNorm(self.fc3  .weight_g.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

# Define Loihi parameter generator
def genLoihiParams(net):
    fcWeights   = []
    convWeights = []
    poolWeights = []
    delay       = []

    loihiWeights = lambda x: snn.quantize.apply(x, 2).flatten().cpu().data.numpy()
    loihiDelays  = lambda d: torch.floor(d).flatten().cpu().data.numpy()

    torch.nn.utils.remove_weight_norm(net.fc1  , name='weight')
    torch.nn.utils.remove_weight_norm(net.fc2  , name='weight')
    torch.nn.utils.remove_weight_norm(net.fc3  , name='weight')

    fcWeights  .append( loihiWeights(net.fc1.weight  ) )
    fcWeights  .append( loihiWeights(net.fc2.weight  ) )
    fcWeights  .append( loihiWeights(net.fc3.weight  ) )

    delay.append( loihiDelays(net.delay1.delay) )
    delay.append( loihiDelays(net.delay2.delay) )
    
    for i in range(len(fcWeights)):     
        np.savetxt('Trained/fc%dWeights.txt'%(i+1), fcWeights[i], fmt='%g')
        plt.figure()
        plt.hist(fcWeights[i], 256)
        plt.title('fc%d weights'%(i+1))
        
    for i in range(len(convWeights)):   
        np.savetxt('Trained/conv%dWeights.txt'%(i+1), convWeights[i], fmt='%g')
        plt.figure()
        plt.hist(convWeights[i], 256)
        plt.title('conv%d weights'%(i+1))
        
    for i in range(len(poolWeights)):   
        np.savetxt('Trained/pool%dWeights.txt'%(i+1), poolWeights[i], fmt='%g')
        plt.figure()
        plt.hist(poolWeights[i], 256)
        plt.title('pool%d weights'%(i+1))

    for i in range(len(delay)):   
        np.savetxt('Trained/delay%d.txt'%(i+1), delay[i], fmt='%g')
        plt.figure()
        plt.hist(delay[i], 64)
        plt.title('delay%d'%(i+1))

        
if __name__ == '__main__':
    netParams = snn.params('network.yaml')
    
    # Define the cuda device to run the code on.
    device = torch.device('cuda:1')
    # deviceIds = [1, 2]

    # Create network instance.
    net = Network(netParams).to(device)
    # net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

    # Create snn loss instance.
    error = snn.loss(netParams, snn.loihi).to(device)

    # Define optimizer module.
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
    optimizer = optim.Nadam(net.parameters(), lr = 0.003)

    # Dataset and dataLoader instances.
    trainingSet = handEmgDataset(samples     =np.arange(675),
                                 samplingTime=netParams['simulation']['Ts'],
                                 sampleLength=netParams['simulation']['tSample'])
    trainLoader = DataLoader(dataset=trainingSet, batch_size=4, shuffle=True, num_workers=1)

    testingSet = handEmgDataset(samples     =np.arange(675, 750),
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=netParams['simulation']['tSample'])
    testLoader = DataLoader(dataset=testingSet, batch_size=4, shuffle=True, num_workers=1)

    # Learning stats instance.
    stats = learningStats()

    lastEpoch = 0
    # lastEpoch = stats.load('Trained/', modulo=10)
    # checkpoint = torch.load('Logs/checkpoint%d.pt'%(lastEpoch -1))
    # net.module.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Visualize the input spikes (first five samples).
    for i in range(5):
        input, target, label = trainingSet[i]
        print(actions[label])
        snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((1, 32, -1)).cpu().data.numpy()), repeat=True)
        
    for epoch in range(lastEpoch, 2000):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            net.train()

            # Move the input and target to correct GPU.
            input  = input.to(device)
            target = target.to(device) 

            # Forward pass of the network.
            output, count = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Clamp delays
            net.clamp()
            # net.module.clamp()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(
                epoch, i, (datetime.now() - tSt).total_seconds(), 
                header=['Spike Count:' + str(torch.sum(count, dim=0).tolist())],
            )

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
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
                epoch, i, 
                header=['Spike Count:' + str(torch.sum(count, dim=0).tolist())],
            )


        # Update stats.
        stats.update()
        stats.plot(saveFig=True, path='Trained/')
        net.gradFlow(path='Trained/')
        # net.module.gradFlow(path='Trained/')
        if stats.testing.bestAccuracy is True:  torch.save(net.state_dict(), 'Trained/emgGesture.pt')
        # if stats.testing.bestAccuracy is True:  torch.save(net.module.state_dict(), 'Trained/emgGesture.pt')
        
        if epoch%10 == 0:
            torch.save(
                {
                    'net': net.state_dict(),
                    # 'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, 
                'Logs/checkpoint%d.pt'%(epoch)
            )
            
        stats.save('Trained/')

    net.module.load_state_dict(torch.load('Trained/emgGesture.pt'))
    genLoihiParams(net)
    # genLoihiParams(net.module)
