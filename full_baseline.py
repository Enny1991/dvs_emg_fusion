import csv
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Add, Concatenate, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras import backend as K
from scipy.signal import stft
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle as pkl

from utils import Person, analyze, corrections


classes = classes = ['pinky', 'elle', 'yo', 'index', 'thumb']
classes_dict = {'pinky': 0, 'elle': 1, 'yo': 2, 'index': 3, 'thumb': 4}
classes_inv = {v: k for k, v in classes_dict.items()}
crop_path = './dump/img_cropped/'
data_dir = '/Users/enea/Dropbox/Capocaccia2019_Gesture_DVS_Myo/Dataset/EMG/'
VERBOSE = False


def load_emg_dataset():
    subjects = {}
    names = sorted([name for name in os.listdir(data_dir) if "emg" in name])
    for name in names:
        _emg = np.load(data_dir + '{}'.format(name)).astype('float32')
        _ann = np.concatenate([np.array(['none']), np.load(data_dir + '{}'.format(name.replace("emg", "ann")))[:-1]])

        subjects["_".join(name.split("_")[:2])] = Person(name.split("_")[0], _emg, _ann, classes=classes)

        if VERBOSE:
            print("Loaded {}: EMG = [{}] // ANN = [{}]".format("_".join(name.split("_")[:2]), _emg.shape, len(_ann)))
    print("Data Loaded! {} Sessions".format(len(subjects.keys())))

    # separates data in correct trial type
    for name, data in subjects.items():
        for _class in classes:
            _annotation = np.float32(data.ann == _class)
            derivative = np.diff(_annotation) / 1.0
            begins = np.where(derivative == 1)[0]
            ends = np.where(derivative == -1)[0]
            for b, e in zip(begins, ends):
                _trials = data.emg[b:e]
                data.trials[_class].append(_trials)
                data.begs[_class].append(b)
                data.ends[_class].append(e)
    print("Done sorting trials!")

    X_EMG = []
    Y_EMG = []
    SUB_EMG = []
    SES_EMG = []
    TRI_EMG = []

    for name, data in subjects.items():
        for gesture in classes:
            for trial in range(5):
                X_EMG.append(data.trials[gesture][trial])
                Y_EMG.append(classes_dict[gesture])
                SUB_EMG.append(int(name[7:9]))
                SES_EMG.append(int(name[17:19]))
                TRI_EMG.append(trial)

    X_EMG = np.array(X_EMG)
    Y_EMG = np.array(Y_EMG)
    SUB_EMG = np.array(SUB_EMG)
    SES_EMG = np.array(SES_EMG)
    TRI_EMG = np.array(TRI_EMG)

    return X_EMG, SUB_EMG, SES_EMG, TRI_EMG, Y_EMG

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def load_img_dataset():
    all_frames = [i for i in sorted(os.listdir(crop_path)) if 'tiff' in i]
    Y_IMG = np.array([classes_dict[i.split('_')[2]] for i in sorted(os.listdir(crop_path)) if 'tiff' in i])
    SUB_IMG = np.array([int(i[7:9]) for i in sorted(os.listdir(crop_path)) if 'tiff' in i])
    SES_IMG = np.array([int(i[17:19]) for i in sorted(os.listdir(crop_path)) if 'tiff' in i])
    TRI_IMG = np.array([int(i.split('_')[3]) for i in sorted(os.listdir(crop_path)) if 'tiff' in i])
    IDX_IMG = np.array([int(i.split('_')[4].split('.')[0]) for i in sorted(os.listdir(crop_path)) if 'tiff' in i])

    IDX_IMG = np.array(IDX_IMG)
    Y_IMG = np.array(Y_IMG)
    SUB_IMG = np.array(SUB_IMG)
    SES_IMG = np.array(SES_IMG)
    TRI_IMG = np.array(TRI_IMG)

    X_IMG = np.array([rgb2gray(plt.imread(crop_path + f))[::-1] for f in all_frames])
    X_IMG = np.array(X_IMG)

    return X_IMG, SUB_IMG, SES_IMG, TRI_IMG, Y_IMG

def sync_datasets(EMG, IMG, frame_len=0.2, frame_step=0.1):
    X_EMG, SUB_EMG, SES_EMG, TRI_EMG, Y_EMG = EMG
    X_IMG, SUB_IMG, SES_IMG, TRI_IMG, Y_IMG = IMG
    F_SUB = []
    F_SESS = []
    F_Y = []
    F_IMG = []
    F_EMG = []

    # CREATE ACTUAL DATASET
    for subject in range(1, 22):
        for session in range(1, 4):
            for gesture in range(5):
                for trial in range(5):
                    fs = corrections['subject{:02}_session0{}'.format(subject, session)]['fs']

                    idx_emg = np.logical_and.reduce([SUB_EMG == subject,
                                                     SES_EMG == session,
                                                     TRI_EMG == trial,
                                                     Y_EMG == gesture])
                    idx_img = np.logical_and.reduce([SUB_IMG == subject,
                                                     SES_IMG == session,
                                                     TRI_IMG == trial,
                                                     Y_IMG == gesture])

                    e = X_EMG[idx_emg][0]
                    f = X_IMG[idx_img]

                    mav = analyze(e, fs=fs, frame_len=frame_len, frame_step=frame_step, feat='MSV', preprocess=False)
                    rms = analyze(e, fs=fs, frame_len=frame_len, frame_step=frame_step, feat='RMS', preprocess=False)
                    a = np.concatenate([mav, rms], 1)

                    mapping = np.arange(len(a)) * len(f) // len(a)
                    F_EMG.append(a)
                    F_IMG.append(np.stack(f[mapping]))
                    F_SUB.append(np.ones((len(mapping))) * subject)
                    F_SESS.append(np.ones((len(mapping))) * session)
                    F_Y.append(np.ones((len(mapping))) * gesture)

    F_EMG = np.vstack(F_EMG)
    F_IMG = np.vstack(F_IMG)
    F_SUB = np.hstack(F_SUB)
    F_SESS = np.hstack(F_SESS)
    F_Y = np.hstack(F_Y)

    return F_EMG, F_IMG, F_SUB, F_SESS, F_Y


def create_sets(F_EMG, F_IMG, F_SESS, F_Y, test_ses=1):

    x_emg_train = F_EMG[F_SESS != test_ses].astype('float32')
    x_emg_test = F_EMG[F_SESS == test_ses].astype('float32')

    x_img_train = F_IMG[F_SESS != test_ses].astype('float32')
    x_img_test = F_IMG[F_SESS == test_ses].astype('float32')

    y_train = F_Y[F_SESS != test_ses]
    y_test = F_Y[F_SESS == test_ses]

    y_train = keras.utils.to_categorical(y_train, 5)
    y_test = keras.utils.to_categorical(y_test, 5)

    return x_emg_train, x_img_train, y_train, x_emg_test, x_img_test, y_test


def preprocess(x_emg_train, x_img_train, x_emg_test, x_img_test):
    # img preprocessing
    # normalize
    data_max = np.max(x_img_train)
    data_min = np.min(x_img_train)
    for i in range(len(x_img_train)):
        x_img_train[i] = (x_img_train[i] - data_min) / (data_max - data_min)
    for i in range(len(x_img_test)):
        x_img_test[i] = (x_img_test[i] - data_min) / (data_max - data_min)

    # standardize
    data_mean = np.mean(x_img_train)
    data_std = np.std(x_img_train) + 1e-15
    x_img_train -= data_mean
    x_img_train /= data_std
    x_img_test -= data_mean
    x_img_test /= data_std

    # emg preprocessing
    # normalize
    data_max = np.max(x_emg_train)
    data_min = np.min(x_emg_train)
    for i in range(len(x_emg_train)):
        x_emg_train[i] = (x_emg_train[i] - data_min) / (data_max - data_min)
    for i in range(len(x_emg_test)):
        x_emg_test[i] = (x_emg_test[i] - data_min) / (data_max - data_min)

    # standardize
    data_mean = np.mean(x_emg_train)
    data_std = np.std(x_emg_train) + 1e-15
    x_emg_train -= data_mean
    x_emg_train /= data_std
    x_emg_test -= data_mean
    x_emg_test /= data_std

    return x_emg_train, x_img_train, x_emg_test, x_img_test


def baseline_charlotte(x_emg_train, x_img_train, y_train, x_emg_test, x_img_test, y_test, n_epochs=(10, 30, 10)):
    # x_img_train_c = np.mean(x_img_train, -1)
    # x_img_test_c = np.mean(x_img_test, -1)
    x_img_train_c = x_img_train
    x_img_test_c = x_img_test

    a = x_img_train_c[:, ::2, ::2]
    b = x_img_train_c[:, 1::2, ::2]
    c = x_img_train_c[:, ::2, 1::2]
    d = x_img_train_c[:, 1::2, 1::2]

    concat_inp_train = [a.reshape(-1, 400), b.reshape(-1, 400), c.reshape(-1, 400), d.reshape(-1, 400)]

    a = x_img_test_c[:, ::2, ::2]
    b = x_img_test_c[:, 1::2, ::2]
    c = x_img_test_c[:, ::2, 1::2]
    d = x_img_test_c[:, 1::2, 1::2]

    concat_inp_test = [a.reshape(-1, 400), b.reshape(-1, 400), c.reshape(-1, 400), d.reshape(-1, 400)]

    # mlp for img
    img_input_shape_c = (400,)
    models = []

    for i in range(4):
        # create the mlp model
        model_img_c = Sequential()
        model_img_c.add(Dense(210, activation='relu', input_shape=img_input_shape_c))
        model_img_c.add(Dropout(0.3))
        model_img_c.add(Dense(5, activation='softmax'))
        models.append(model_img_c)

    # compile model
    for inp_train, inp_test, mdl in zip(concat_inp_train, concat_inp_test, models):
        mdl.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        # fit the model
        mdl.fit(inp_train, y_train,
                batch_size=32,
                epochs=n_epochs[0],
                verbose=1,
                validation_data=(inp_test, y_test))

    all_scores = []
    for inp_test, mdl in zip(concat_inp_test, models):
        # evaluate the model
        all_scores.append(mdl.predict(inp_test))
    # evaluate the model
    final = np.argmax(sum(all_scores), -1)
    final_y = np.argmax(y_test, -1)

    score_img_c = np.sum(np.float32((final_y - final) == 0)) / len(final_y) * 100

    # mlp for emg
    img_input_shape_c = (16,)
    # create the cnn model
    model_emg_c = Sequential()
    model_emg_c.add(Dense(230, activation='relu', input_shape=img_input_shape_c))
    model_emg_c.add(Dense(5, activation='softmax'))

    # compile model
    model_emg_c.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    # fit the model
    model_emg_c.fit(x_emg_train, y_train,
                    batch_size=32,
                    epochs=n_epochs[1],
                    verbose=1,
                    validation_data=(x_emg_test, y_test))

    # evaluate the model
    scores = model_emg_c.evaluate(x_emg_test, y_test, verbose=0)
    score_emg_c = scores[1] * 100

    mergedOut_c = Concatenate()([a.output for a in models] + [model_emg_c.output])
    mergedOut_c = Dense(5, activation='softmax')(mergedOut_c)
    model_fus_c = Model([a.input for a in models] + [model_emg_c.input], mergedOut_c)

    for layer in model_fus_c.layers[:len(model_fus_c.layers) - 1]:
        layer.trainable = False

    # compile model
    model_fus_c.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    # fit the model
    model_fus_c.fit(concat_inp_train + [x_emg_train], y_train,
                    batch_size=32,
                    epochs=n_epochs[2],
                    verbose=1,
                    validation_data=(concat_inp_test + [x_emg_test], y_test))

    # evaluate the model
    scores = model_fus_c.evaluate(concat_inp_test + [x_emg_test], y_test, verbose=1)
    score_fus_c = scores[1] * 100

    return score_emg_c, score_img_c, score_fus_c


def baseline_sumit(x_emg_train, x_img_train, y_train, x_emg_test, x_img_test, y_test, n_epochs=(10, 30, 10)):
    # cnn for img
    img_input_shape_s = (40, 40, 1)
    num_classes = 5
    # create the cnn model
    model_img_s = Sequential()
    model_img_s.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=img_input_shape_s))
    model_img_s.add(MaxPooling2D(pool_size=(2, 2)))
    model_img_s.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model_img_s.add(MaxPooling2D(pool_size=(2, 2)))
    model_img_s.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model_img_s.add(Dropout(0.25))
    model_img_s.add(Flatten())
    model_img_s.add(Dense(512, activation='relu'))
    model_img_s.add(Dropout(0.5))
    model_img_s.add(Dense(num_classes, activation='softmax'))

    # compile model
    model_img_s.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    # fit the model
    model_img_s.fit(x_img_train, y_train,
            batch_size=32,
            epochs=n_epochs[0],
            verbose=1,
            validation_data=(x_img_test, y_test))

    # evaluate the model
    scores = model_img_s.evaluate(x_img_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model_img_s.metrics_names[1], scores[1] * 100))
    score_img_s = scores[1] * 100

    # mlp for emg
    img_input_shape_s = (16,)
    # create the cnn model
    model_emg_s = Sequential()
    model_emg_s.add(Dense(128, activation='relu', input_shape=img_input_shape_s))
    model_emg_s.add(Dropout(0.5))
    model_emg_s.add(Dense(128, activation='relu'))
    model_emg_s.add(Dense(num_classes, activation='softmax'))

    # compile model
    model_emg_s.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

    # fit the model
    model_emg_s.fit(x_emg_train, y_train,
                    batch_size=32,
                    epochs=n_epochs[1],
                    verbose=1,
                    validation_data=(x_emg_test, y_test))

    # evaluate the model
    scores = model_emg_s.evaluate(x_emg_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model_emg_s.metrics_names[1], scores[1] * 100))

    score_emg_s = scores[1] * 100

    mergedOut_s = Concatenate()([model_img_s.output, model_emg_s.output])
    mergedOut_s = Dense(5, activation='softmax')(mergedOut_s)
    model_fus_s = Model([model_img_s.input, model_emg_s.input], mergedOut_s)

    # freeze the layers except the last dense
    for layer in model_fus_s.layers[:len(model_fus_s.layers) - 1]:
        layer.trainable = False

    # compile model
    model_fus_s.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    # fit the model
    model_fus_s.fit([x_img_train, x_emg_train], y_train,
                    batch_size=32,
                    epochs=n_epochs[2],
                    verbose=1,
                    validation_data=([x_img_test, x_emg_test], y_test))

    # evaluate the model
    scores = model_fus_s.evaluate([x_img_test, x_emg_test], y_test, verbose=1)
    print("%s: %.2f%%" % (model_fus_s.metrics_names[1], scores[1] * 100))

    score_fus_s = scores[1] * 100

    return score_emg_s, score_img_s, score_fus_s


def main():

    EMG = load_emg_dataset()
    IMG = load_img_dataset()

    frame_lens = [0.2, 0.15, 0.1, 0.070, 0.050, 0.030, 0.015]
    n_epochs = [(10, 15, 10), (10, 15, 10), (8, 12, 8), (7, 11, 7), (5, 10, 5), (3, 10, 3), (2, 10, 2)]

    with open('full_results_cross_latency_v3_grayscale.csv', 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        header = ["model", "feat", "frame_len", "test_ses", "accuracy"]
        file_writer.writerow(header)

        for test_ses in [1, 2, 3]:
            for frame_len, n_epoch in zip(frame_lens, n_epochs):
                print(f"Doing {test_ses} and {frame_len}")
                F_EMG, F_IMG, F_SUB, F_SESS, F_Y = sync_datasets(EMG, IMG, frame_len=frame_len, frame_step=frame_len / 2)

                x_emg_train, x_img_train, y_train, x_emg_test, x_img_test, y_test = create_sets(F_EMG, F_IMG, F_SESS, F_Y,
                                                                                                test_ses=test_ses)

                x_emg_train, x_img_train, x_emg_test, x_img_test = preprocess(x_emg_train, x_img_train,
                                                                              x_emg_test, x_img_test)

                score_emg_s, score_img_s, score_fus_s = baseline_sumit(x_emg_train, x_img_train[..., None], y_train,
                                                                       x_emg_test, x_img_test[..., None], y_test, n_epochs=n_epoch)

                score_emg_c, score_img_c, score_fus_c = baseline_charlotte(x_emg_train, x_img_train, y_train,
                                                                           x_emg_test, x_img_test, y_test, n_epochs=n_epoch)

                file_writer.writerow(['sumit', "emg", frame_len, test_ses, score_emg_s])
                file_writer.writerow(['sumit', "img", frame_len, test_ses, score_img_s])
                file_writer.writerow(['sumit', "fus", frame_len, test_ses, score_fus_s])
                file_writer.writerow(['charlotte', "emg", frame_len, test_ses, score_emg_c])
                file_writer.writerow(['charlotte', "img", frame_len, test_ses, score_img_c])
                file_writer.writerow(['charlotte', "fus", frame_len, test_ses, score_fus_c])


if __name__ == '__main__':
    main()
