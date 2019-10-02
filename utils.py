import numpy as np
import scipy as sc
from scipy.signal import butter, lfilter, welch, square  #for signal filtering
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os
from os import listdir


def find_trigger(ts):
    return np.where(np.diff(ts) < 0)[0][0]


def create_frame(x, y, dim=(128, 128)):
    img = np.zeros(dim)
    for _x, _y in zip(x.astype('int32'), y.astype('int32')):
        img[dim[0] - 1 - _x,_y] += 1
    img /= np.max(img) + 1e-15
    return np.uint8(img * 255) 

def norm_frame(x):
    x = np.float32(x)
    x -= np.min(x, (1, 2), keepdims=True)
    x /= np.max(x, (1, 2), keepdims=True) + 1e-15
    return np.uint8(x * 255)


def train_svm_single(x_train, y_train, kernel='linear', save_path=None):
    clf = SVC(gamma='auto', kernel=kernel)
    clf.fit(x_train, y_train)
    _acc_train = clf.score(x_train, y_train)
    if save_path is not None:
        clf.save(save_path)
    return _acc_train, clf


def do_tc_single(x_train, y_train, x_test, y_test, kernel='linear'):
    clf = SVC(gamma='auto', kernel=kernel)
    clf.fit(x_train, y_train)
    _acc_test = clf.score(x_test, y_test)
    _acc_train = clf.score(x_train, y_train)
    return _acc_train, _acc_test


def do_tc_full(x, y, pca_comp=0, shuffle=True, folds=10, verbose=False, avg=False, kernel='linear'):
    
    n_samples, dims = x.shape
    
    if pca_comp > 0 :
        keep = pca_comp
        pca = PCA(n_components=keep)
        x = pca.fit_transform(x)
        if verbose:
            print("Features: {} => {}".format(dims, x.shape[1]))
    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    
    all_acc_train = []
    all_acc_test = []
    split = n_samples // folds
    
    if verbose:
        print("Train on: {} / Test on {}".format(n_samples - split, split))

    for fold in range(folds):

        test_X = x[fold * split: (fold + 1) * split]
        train_X = np.vstack([x[:split * fold], x[split * (fold + 1):]])

        test_y = y[fold * split: (fold + 1) * split].squeeze()
        train_y = np.vstack([y[:split * fold], y[split * (fold + 1):]]).squeeze()

        _acc_train, _acc_test = do_tc_single(train_X, train_y, test_X, test_y, kernel=kernel)
        
        all_acc_train.append(_acc_train)
        all_acc_test.append(_acc_test)
        
        if verbose:
            print("Fold {}: Train acc: {:.3} || Test acc: {:.3}".format(fold, _acc_train, _acc_test))
    if avg:
        return np.mean(all_acc_train), np.mean(all_acc_test), np.std(all_acc_train), np.std(all_acc_test)
    else:
        return all_acc_train, all_acc_test
    

def kfold(x, y, folds=5):
    
    n_samples = len(x)
    
    idx = np.random.permutation(n_samples)
    x = x[idx]
    y = y[idx]
    
    split = n_samples // folds
    FOLDS = []

    for fold in range(folds):

        test_x = x[fold * split: (fold + 1) * split]
        train_x = np.vstack([x[:split * fold], x[split * (fold + 1):]])

        test_y = y[fold * split: (fold + 1) * split].squeeze()
        train_y = np.vstack([y[:split * fold], y[split * (fold + 1):]]).squeeze()
        
        FOLDS.append([train_x, train_y, test_x, test_y])
    return FOLDS


#Define the filters
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return(b, a)
    
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return(b, a)
    
    
def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return(y)
    
    
def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return(y)


def analyze(x, fs=200, frame_len=0.3, frame_step=0.1, feat='MAV', order=1, threshold=0.1, preprocess=True):
    l, n_ch = x.shape
    frame_len_samples = int(frame_len * fs)
    frame_step_samples = int(frame_step * fs)
    
    n_win = (l - frame_len_samples) // frame_step_samples + 1
    ret = np.zeros((n_win, n_ch))
    
    w = np.hamming(frame_len_samples)
    
    if preprocess:
        filteredEMGSignal = butter_lowpass_filter(x.T, 99, fs, 2)
        x = butter_highpass_filter(filteredEMGSignal, 20, fs, 2).T

        
    for i in range(n_win):
        chunk = x[i * frame_step_samples: i * frame_step_samples + frame_len_samples]
        
        # TIME DOMAIN
        if feat == 'MAV':
            ret[i] = np.mean(np.abs(chunk), 0)
        if feat == 'iEMG':
            ret[i] = np.sum(np.abs(chunk), 0)
        if feat == 'SSI':
            ret[i] = np.sum(chunk ** 2, 0)
        if feat == 'VAR':
            ret[i] = np.sum(chunk ** 2, 0) / (frame_len_samples - 1)
        if feat == 'SD':
            ret[i] = np.std(chunk, 0)
        if feat == 'TM':
            ret[i] = np.abs(np.sum(chunk ** order, 0) / frame_len_samples)
        if feat == 'RMS':
            ret[i] = np.sqrt(np.sum(chunk ** 2, 0) / frame_len_samples)
        if feat == 'LOG':
            ret[i] = np.exp(np.sum(np.abs(chunk), 0) / frame_len_samples) 
        if feat == 'WL':
            ret[i] = np.sum(np.abs(np.diff(chunk.T).T), 0)
        if feat == 'AAC':
            ret[i] = np.sum(np.abs(np.diff(chunk.T).T), 0) / frame_len_samples
        if feat == 'DASDV':
            ret[i] = np.sum(np.diff(chunk.T).T ** 2, 0) / (frame_len_samples - 1)
        if feat == 'AFB':
            for j in range(n_ch):
                _s = np.convolve(w / w.sum(), chunk[:, j] ** 2, mode='valid')
                pp = peakutils.indexes(_s)
                if len(pp) > 0:
                    peak = pp[0]
                else:
                    peak = 0
                ret[i, j] = _s[peak]
        if feat == 'ZC':
            positive = [chunk[0, i] > threshold for i in range(n_ch)]
            ZC = np.zeros((n_ch,))
            for j in range(n_ch):
                for ss in chunk[1:, j]:
                    if positive[j]:
                        if(ss < 0 -threshold):
                            positive[j] = False
                            ZC[j] += 1
                    else:
                        if(ss > 0 + threshold):
                            positive[j] = True
            ret[i] = ZC
        if feat == 'MYOP':
            ret[i] = np.sum(np.float32(chunk >= threshold), 0) / frame_len_samples
        if feat == 'WAMP':
            ret[i] = np.sum(np.float32(np.diff(chunk.T).T > threshold))
        if feat == 'SSC':
            N = len(chunck)
            SSC = 0
            for i in range(1,N-1):
                a, b, c = [chunk[i-1], chunk[i], chunk[i+1]]
                if(a + b + c >= threshold * 3 ): #computed only if the 3 values are above the threshold
                    if(a < b > c or a > b < c ): #if there's change in the slope
                        SSC += 1
            ret[i] = SSC
            
        # FREQ DOMAIN
        
    return ret




def window_spikes(ts, ch, w=0.005, limit=False, n_ch=None):
    
    if n_ch is None:
        n_ch = len(set(ch))
    
    ts -= np.min(ts)
    ts_int = (ts // w).astype('int32')
    
    if len(ts_int) < 1:
        ts_int = np.array([0,0])

    A = np.zeros((np.max(ts_int) + 1, n_ch))

    for _t, _c in zip(ts_int, ch):
        A[_t, _c] += 1
        
    if limit:
        A = np.minimum(A, np.ones_like(A))
    return A


def simple_low_pass(X, win=12, shift=1):
    # X (time, channels)
    if len(X.shape) == 1:
        X = np.expand_dims(X, 1)
    X = np.vstack([X, np.zeros((win, X.shape[1]))])
    n_win = (X.shape[0] - win) // shift
    XX = np.zeros((n_win, X.shape[1]))
    for i in range(0, n_win):
        XX[i] = np.mean(X[i * shift: i * shift + win] , 0) 
    return np.squeeze(XX)


def exp_feat(A, win=0.05, tpe='lap', l=300):
    if tpe == 'exp':
        t = np.arange(0, win, 0.001)
        b = np.exp(-l * t)
    elif tpe == 'lap':
        t = np.arange(-win, win, 0.001)
        b = np.exp(-l * np.abs(t))
        
    b /= np.linalg.norm(b)
    AA = np.array([np.convolve(_a, b, 'same') for _a in A])
    return AA


def signal_to_spike_refractory(interpfact, time, amplitude, thr_up, thr_dn, refractory_period):
    actual_dc = 0
    spike_up = []
    spike_up = []
    spike_dn = []
    last_sample = interpfact * refractory_period

    f = sc.interpolate.interp1d(time, amplitude)
    rangeint = np.round((np.max(time) - np.min(time)) * interpfact)
    xnew = np.linspace(np.min(time), np.max(time), num=int(rangeint), endpoint=True)
    data = np.reshape([xnew, f(xnew)], (2, len(xnew))).T

    i = 0
    while i < (len(data) - int(last_sample)):
        if ((actual_dc + thr_up) < data[i, 1]):
            spike_up.append(data[i, 0])  # spike up
            actual_dc = data[i, 1]  # update current dc value
            i += int(refractory_period * interpfact)
        elif ((actual_dc - thr_dn) > data[i, 1]):
            spike_dn.append(data[i, 0])  # spike dn
            actual_dc = data[i, 1]  # update curre
            i += int(refractory_period * interpfact)
        else:
            i += 1

    return spike_up, spike_dn


class Person(object):
    def __init__(self, name, emg, ann, classes=['rock', 'paper', 'scissor']):
        self.name = name
        self.emg = emg
        self.ann = ann
        self.trials = {c: [] for c in classes}
        self.begs = {c: [] for c in classes}
        self.ends = {c: [] for c in classes}
        self.x = {c: [] for c in classes}
        self.y = {c: [] for c in classes}
        self.ts = {c: [] for c in classes}
        self.pol = {c: [] for c in classes}
        self.emg_spks = {c: [] for c in classes}

        self.frames = {c: [] for c in classes}
        self.ts_frames = {c: [] for c in classes}
        self.x_davis = {c: [] for c in classes}
        self.y_davis = {c: [] for c in classes}
        self.ts_davis = {c: [] for c in classes}
        self.pol_davis = {c: [] for c in classes}
        
        self.spk_trials = {c: [] for c in classes}
#         self.trials = {'rock': [], 'paper': [], 'scissor': []}
#         self.spk_trials = {'rock': [], 'paper': [], 'scissor': []}
       
     
def load_all_emg(data_dir, classes, verbose=False):
    subjects = {}
    names = [name for name in listdir(data_dir) if "emg" in name]
    for name in names:
        _emg = np.load(data_dir + '{}'.format(name)).astype('float32')
        _ann = np.concatenate([np.array(['none']), np.load(data_dir + '{}'.format(name.replace("emg","ann")))[:-1]])

        subjects["_".join(name.split("_")[:2])] = Person(name.split("_")[0], _emg, _ann, classes=classes)

        if verbose:
            print("Loaded {}: EMG = [{}] // ANN = [{}]".format("_".join(name.split("_")[:2]), _emg.shape, len(_ann)))
    

    # separates data in correct trial type
    for name, data in subjects.items():
        for _class in classes:
            _annotation = np.float32(data.ann == _class)
            derivative = np.diff(_annotation)/1.0
            begins = np.where(derivative == 1)[0]
            ends = np.where(derivative == -1)[0]
            for b, e in zip(begins, ends):
                _trials = data.emg[b:e]
                data.trials[_class].append(_trials)
                data.begs[_class].append(b)
                data.ends[_class].append(e)
                
    print("Data Loaded! {} Sessions".format(len(subjects.keys())))
    return subjects

# this is to find the zero timestepping of the dvs
def find_trigger(ts):
    return np.where(np.diff(ts) < 0)[0][0]