import numpy as np

samples = np.random.permutation(1575)
np.savetxt('train.txt', samples[:1417], fmt='%g')
np.savetxt('test.txt' , samples[1417:], fmt='%g')