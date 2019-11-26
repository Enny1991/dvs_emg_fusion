from datetime import datetime
import pickle as pkl
import numpy as np
import matplotlib
if __name__ == '__main__': 
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import slayerSNN.optimizer as optim
from slayerSNN.learningStats import learningStats

def augmentData(xEvent, yEvent):
    xs = 4
    ys = 4
    th = 10
    scaleMin = 0.8
    scaleMax = 2
    xjitter = np.random.randint(2*xs) - xs
    yjitter = np.random.randint(2*ys) - ys
    ajitter = (np.random.rand()-0.5) * th / 180 * 3.141592654
    sjitter = np.random.randint(scaleMax - scaleMin) + scaleMin
    sinTh = np.sin(ajitter)
    cosTh = np.cos(ajitter)
    xEvent = xEvent * cosTh - yEvent * sinTh + xjitter
    yEvent = xEvent * sinTh + yEvent * cosTh + yjitter
    xEvent = (xEvent - 20) * sjitter + 20
    yEvent = (yEvent - 20) * sjitter + 20
    return xEvent, yEvent

class fusion(Dataset):
    def __init__(self, samples, samplingTime, sampleLength, augment=False):
        self.data         = pkl.load(open('../data/relax21_cropped_dvs_emg_spikes.pkl', 'rb'))
        self.samples      = samples
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)
        self.augment      = augment

    def __getitem__(self, index):
        # load event and label
        sampleID = self.samples[index]
        label = int(self.data['y'][sampleID])
        event = snn.io.readNpSpikes('fusedFeatures/{}.npy'.format(sampleID))

        input = event.toSpikeTensor(
            torch.zeros((512+128, 1, 1,self.nTimeBins)),
            samplingTime=self.samplingTime
        )

        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((5, 1, 1, 1))
        desiredClass[label,...] = 1

        return input, desiredClass, label

    def __len__(self):
        return self.samples.shape[0]

class fusionDataset(Dataset):
    def __init__(self, samples, samplingTime, sampleLength, augment=False):
        self.data         = pkl.load(open('../data/relax21_cropped_dvs_emg_spikes.pkl', 'rb'))
        self.samples      = samples
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)
        self.augment      = augment

    def __getitem__(self, index):
        # load event and label
        sampleID = self.samples[index]
        label = int(self.data['y'][sampleID])
        emgEvent = self.data['emg'][sampleID]
        dvsEvent = self.data['dvs'][sampleID]

        if self.augment is True:    
            dvsEvent[0], dvsEvent[1] = augmentData(dvsEvent[0], dvsEvent[1])

        emgEvent = snn.io.event(emgEvent[0], None, emgEvent[2], emgEvent[1]*1000)
        dvsEvent = snn.io.event(dvsEvent[0], dvsEvent[1], dvsEvent[3], dvsEvent[2]*1000)

        # random shift
        tSt = 0 
        if self.nTimeBins < 2000:
            tSt = np.random.randint(2000 - self.nTimeBins)

            ind = (emgEvent.t >= tSt) & (emgEvent.t < tSt + self.nTimeBins)
            emgEvent.x = emgEvent.x[ind]
            emgEvent.p = emgEvent.p[ind]
            emgEvent.t = emgEvent.t[ind] - tSt

            ind = (dvsEvent.t >= tSt) & (dvsEvent.t < tSt + self.nTimeBins)
            dvsEvent.x = dvsEvent.x[ind]
            dvsEvent.y = dvsEvent.y[ind]
            dvsEvent.p = dvsEvent.p[ind]
            dvsEvent.t = dvsEvent.t[ind] - tSt

        emgInput = emgEvent.toSpikeTensor(
            torch.zeros((2, 1, 8,self.nTimeBins)),
            samplingTime=self.samplingTime
        )

        dvsInput = dvsEvent.toSpikeTensor(
            torch.zeros((2,40,40,self.nTimeBins)),
            samplingTime=self.samplingTime
        )
        
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((5, 1, 1, 1))
        desiredClass[label,...] = 1

        return emgInput, dvsInput, desiredClass, label, sampleID

    def __len__(self):
        return self.samples.shape[0]

class emgFeature(torch.nn.Module):
    def __init__(self, path):
        super(emgFeature, self).__init__()
        netParams = snn.params(path + '/network.yaml')
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.fc1   = slayer.dense( 16, 128)
        self.fc2   = slayer.dense(128, 128)
        self.fc3   = slayer.dense(128,   5)
        self.delay1= slayer.delay(128)
        self.delay2= slayer.delay(128)

        self.fc1.weight.data = torch.FloatTensor(np.load(path + '/Trained/fc1Weights.npy').reshape(self.fc1.weight.shape))
        self.fc2.weight.data = torch.FloatTensor(np.load(path + '/Trained/fc2Weights.npy').reshape(self.fc2.weight.shape))
        self.fc3.weight.data = torch.FloatTensor(np.load(path + '/Trained/fc3Weights.npy').reshape(self.fc3.weight.shape))

        self.delay1.delay.data = torch.FloatTensor(np.load(path + '/Trained/delay1.npy').reshape(self.delay1.delay.shape))
        self.delay2.delay.data = torch.FloatTensor(np.load(path + '/Trained/delay2.npy').reshape(self.delay2.delay.shape))
        
    def forward(self, spike):
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1])) # 32
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 128
        spike = self.slayer.delayShift(spike, 1)

        spike = self.delay1(spike)
        spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 128
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.delay2(spike)
        # spike = self.slayer.spikeLoihi(self.fc3  (spike)) # 128
        # spike = self.slayer.delayShift(spike, 1)

        return spike

class dvsFeature(torch.nn.Module):
    def __init__(self, path):
        super(dvsFeature, self).__init__()
        netParams = snn.params(path + '/network.yaml')
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv( 2,  8, 3, padding=1, weightScale=20)
        self.conv2 = slayer.conv( 8, 16, 3, padding=1, weightScale=100)
        self.conv3 = slayer.conv(16, 32, 3, padding=1, weightScale=100)
        self.fc1   = slayer.dense((10*10*32), 512)
        self.fc2   = slayer.dense(512, 5)
        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)

        self.fc1  .weight.data = torch.FloatTensor(np.load(path + '/Trained/fc1Weights.npy'  ).reshape(self.fc1  .weight.shape))
        self.fc2  .weight.data = torch.FloatTensor(np.load(path + '/Trained/fc2Weights.npy'  ).reshape(self.fc2  .weight.shape))
        self.conv1.weight.data = torch.FloatTensor(np.load(path + '/Trained/conv1Weights.npy').reshape(self.conv1.weight.shape))
        self.conv2.weight.data = torch.FloatTensor(np.load(path + '/Trained/conv2Weights.npy').reshape(self.conv2.weight.shape))
        self.conv3.weight.data = torch.FloatTensor(np.load(path + '/Trained/conv3Weights.npy').reshape(self.conv3.weight.shape))
        self.pool1.weight.data = torch.FloatTensor(np.load(path + '/Trained/pool1Weights.npy').reshape(self.pool1.weight.shape))
        self.pool2.weight.data = torch.FloatTensor(np.load(path + '/Trained/pool2Weights.npy').reshape(self.pool2.weight.shape))

    def forward(self, spike):
        spike = self.slayer.spikeLoihi(self.conv1(spike)) # 128, 128, 8
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool1(spike)) # 64, 64, 8
        spike = self.slayer.delayShift(spike, 1)

        spike = self.slayer.spikeLoihi(self.conv2(spike)) # 64, 64, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool2(spike)) # 32, 32, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.conv3(spike)) # 32, 32, 32
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        spike = self.slayer.delayShift(spike, 1)

        # spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 11
        # spike = self.slayer.delayShift(spike, 1)
        
        return spike

class Network(torch.nn.Module):
    def __init__(self, netParams, emgPath, dvsPath):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.fc1   = torch.nn.utils.weight_norm(slayer.dense(512 + 128, 5), name='weight')

        self.emgNet = emgFeature(emgPath)
        self.dvsNet = dvsFeature(dvsPath)

    def forward(self, emgSpike, dvsSpike):
        emgSpike = self.emgNet.forward(emgSpike)
        dvsSpike = self.dvsNet.forward(dvsSpike)
        spike = torch.cat((emgSpike, dvsSpike), dim=1).detach() # concatenate in the neuron dimension
        # spike = dvsSpike

        # spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        # spike = self.slayer.delayShift(spike, 1)
        
        return spike

class fusionNet(torch.nn.Module):
    def __init__(self, netParams):
        super(fusionNet, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.fc1   = slayer.dense(512 + 128, 5)
        self.fc1.weight.data = torch.FloatTensor(np.load('Trained/fc1Weights.npy').reshape(self.fc1.weight.shape))

    def forward(self, spike):
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        spike = self.slayer.delayShift(spike, 1)
        
        return spike

# Define Loihi parameter generator
def genLoihiParams(net):
    fcWeights   = []

    loihiWeights = lambda x: snn.utils.quantize(x, step=2).cpu().data.numpy()
    
    torch.nn.utils.remove_weight_norm(net.fc1  , name='weight')
    
    fcWeights  .append( loihiWeights(net.fc1.weight  ) )
    
    for i in range(len(fcWeights)):     
        np.save('Trained/fc{:d}Weights.npy'.format(i+1), fcWeights[i].squeeze())
        plt.figure()
        plt.hist(fcWeights[i].flatten(), 256)
        plt.title('fc{:d} weights'.format(i+1))

if __name__ == '__main__':

    netParams = snn.params('network.yaml')
    
    # Define the cuda device to run the code on.
    device = torch.device('cuda')
    # deviceIds = [1, 2]

    # Create network instance.
    net = Network(netParams, 'emg', 'dvsCropped').to(device)

    fus = fusionNet(netParams).to(device)
    # net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

    # Create snn loss instance.
    error = snn.loss(netParams, snn.loihi).to(device)

    # Define optimizer module.
    optimizer = optim.Nadam(net.parameters(), lr = 0.01)

    # Dataset and dataLoader instances.
    trainingSet = fusionDataset(
        samples     =np.loadtxt('train.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=2000,
    )

    testingSet = fusionDataset(
        samples     =np.loadtxt('test.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=2000,
    )

    trainLoader = DataLoader(dataset=trainingSet, batch_size=1, shuffle=False, num_workers=1)
    testLoader  = DataLoader(dataset=testingSet , batch_size=1, shuffle=False, num_workers=1)

    # # Learning stats instance.
    # stats = learningStats()

    # lastEpoch = 0

    # for i, (emgInput, dvsInput, target, label, sample) in enumerate(trainLoader, 0):
    #     net.eval()
    #     with torch.no_grad():
    #         emgInput  = emgInput.to(device)
    #         dvsInput  = dvsInput.to(device)
    #         target    = target.to(device) 

    #         output = net.forward(emgInput, dvsInput)

    #         event = snn.io.spikeArrayToEvent(output.reshape((512 + 128, 1, 1, -1)).cpu().data.numpy())
    #         snn.io.encodeNpSpikes('fusedFeatures/{}.npy'.format(sample.item()), event)

        

    for i, (emgInput, dvsInput, target, label, sample) in enumerate(testLoader, 0):
        net.eval()
        with torch.no_grad():
            emgInput  = emgInput.to(device)
            dvsInput  = dvsInput.to(device)
            target    = target.to(device) 

            output = net.forward(emgInput, dvsInput)

            print(sample.item())
            # event = snn.io.spikeArrayToEvent(output.reshape((512 + 128, 1, 1, -1)).cpu().data.numpy())
            # snn.io.encodeNpSpikes('fusedFeatures/{}.npy'.format(sample.item()), event)


    # Dataset and dataLoader instances.
    trainingSet = fusion(
        samples     =np.loadtxt('train.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=2000,
    )

    testingSet = fusion(
        samples     =np.loadtxt('test.txt').astype(int),
        samplingTime=netParams['simulation']['Ts'],
        sampleLength=2000,
    )

    trainLoader = DataLoader(dataset=trainingSet, batch_size=1, shuffle=True, num_workers=1)
    testLoader  = DataLoader(dataset=testingSet , batch_size=1, shuffle=True, num_workers=1)

    # Learning stats instance.
    stats = learningStats()

    lastEpoch = 0

    for i, (input, target, label) in enumerate(trainLoader, 0):
        net.eval()
        with torch.no_grad():
            input  = input.to(device)
            target = target.to(device) 

            output = fus.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            stats.print(
                0, i
            )

        

    for i, (input, target, label) in enumerate(testLoader, 0):
        net.eval()
        with torch.no_grad():
            input  = input.to(device)
            target = target.to(device) 

            output = fus.forward(input)

            # Gather the training stats.
            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            stats.print(
                0, i
            )

