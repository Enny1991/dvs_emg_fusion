# -*- coding: utf-8 -*-
"""
E2MG results summary for sensor fusion on the ODIN and MorphIC chips.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

def compute_scores (scores, lat_times):
    mean = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)
    for t in range(len(lat_times)):
        print("    After "+str(int(lat_times[t]))+"ms inference: "+str(mean[t])+"% accuracy, with std dev "+str(std[t])+"%")
    return (mean, std)


lat_times = (10, 30, 50, 70, 100, 150, 200)  #milliseconds

#Load results
#   EMG_final_scores: accuracy results for each cross-validation set (3) after 200ms with a 3-bit 16-230-5 MLP on ODIN.
#   DVS_final_scores: accuracy results for each cross-validation set (3) after 200ms with 4x 1-bit 400-210-5 MLPs on MorphIC.
#   fusion_scores:    accuracy results for each cross-validation set (3) after different latencies (7) with the fusion network
#                     (i.e. retrained 3-bit 5-neuron output layer with 4x210+230=1070 inputs)
#   fusion_SOPs:      number of SOPs processed for DVS/MorphIC, EMG/ODIN, Fusion/ODIN for each cross-validation set (3) after different latencies (7).
#                     includes the dummy crossbar SOPs processed for DVS/MorphIC (only 210 out of 512 are actually used for computation)
#                     and the dummy crossbar SOPs processed for EMG/ODIN (only 230 out of 256 are actually used for computation).
#                     All SOPs for the last layer are used for computation, does not account for the energy cost of the external mapping table.
[fusion_scores, fusion_SOPs, fusion_Einf, fusion_Tinf] = pickle.load(open('final_fusion.pkl', 'rb'))
EMG_scores = np.zeros((3,7))
DVS_scores = np.zeros((3,7))
for crossval in range(3):
    EMG_scores[crossval,:] = pickle.load(open('lat_energy_ODIN_cv'+str(crossval)+'.pkl', 'rb'))[0][crossval,:]
    DVS_scores[crossval,:] = pickle.load(open('lat_energy_MorphIC_cv'+str(crossval)+'.pkl', 'rb'))[0][crossval,:]

#Corrections for quad-core PARALLEL processing in MorphIC: inference time is divided by 4, power is multiplied by 4, energy does not change (the activity evenly distributed among the 4 cores)
fusion_Tinf[0,:,:] /= 4
fusion_Tinf[2,:,:] /= 4

#Extract results and write text summary:
print("=== FINAL SUMMARY ===")
print("= DVS SCORES =")
(DVS_mean, DVS_std) = compute_scores(DVS_scores, lat_times)
print("= EMG SCORES =")
(EMG_mean, EMG_std) = compute_scores(EMG_scores, lat_times)
print("= FUSION SCORES =")
(fusion_mean, fusion_std) = compute_scores(fusion_scores, lat_times)

SOP_mean_per_chip = np.mean(fusion_SOPs, axis=1)
SOP_std_per_chip = np.std(fusion_SOPs, axis=1)
SOP_mean = np.mean(np.sum(fusion_SOPs[2:,:,:],axis=0), axis=0)
SOP_std = np.std(np.sum(fusion_SOPs[2:,:,:],axis=0), axis=0)
Einf_mean_per_chip = np.mean(fusion_Einf, axis=1)
Einf_std_per_chip = np.std(fusion_Einf, axis=1)
Einf_mean = np.mean(np.sum(fusion_Einf[2:,:,:],axis=0), axis=0)
Einf_std = np.std(np.sum(fusion_Einf[2:,:,:],axis=0), axis=0)
fusion_Tinf[4,:,] = fusion_Tinf[4,:,] + fusion_Tinf[3,:,]
Tinf_mean_per_chip = np.mean(fusion_Tinf, axis=1)
Tinf_std_per_chip = np.std(fusion_Tinf, axis=1)
Tinf_mean = np.mean(np.max(fusion_Tinf[2:,:,:],axis=0), axis=0)
Tinf_std = np.std(np.max(fusion_Tinf[2:,:,:],axis=0), axis=0)
fusion_Pinf = fusion_Einf*fusion_Tinf
Pinf_mean_per_chip = np.mean(fusion_Pinf, axis=1)
Pinf_std_per_chip = np.std(fusion_Pinf, axis=1)
Pinf_mean = np.mean(np.sum(fusion_Pinf[2:,:,:],axis=0), axis=0)
Pinf_std = np.std(np.sum(fusion_Pinf[2:,:,:],axis=0), axis=0)


#Accuracy/latency plot showing standard deviations
plt.figure()
dvs = plt.errorbar(lat_times, DVS_mean, DVS_std, marker='o')
emg = plt.errorbar(lat_times, EMG_mean, EMG_std, marker='o')
fus = plt.errorbar(lat_times, fusion_mean, fusion_std, marker='o')
plt.legend([dvs,emg,fus], ["DVS data (MorphIC)", "EMG data (ODIN)", "Sensor fusion"], loc="lower right")
plt.xlabel('Inference time [ms]')
plt.ylabel('Inference accuracy [%]')

#SOP/latency plot
plt.figure()
dvs = plt.errorbar(lat_times, SOP_mean_per_chip[0,:]/1e6, SOP_std_per_chip[0,:]/1e6, marker='o')
emg = plt.errorbar(lat_times, SOP_mean_per_chip[1,:]/1e6, SOP_std_per_chip[1,:]/1e6, marker='o')
fus = plt.errorbar(lat_times, SOP_mean/1e6, SOP_std/1e6, marker='o')
plt.legend([dvs,emg,fus], ["DVS data (MorphIC)", "EMG data (ODIN)", "Sensor fusion"], loc="upper left")
plt.xlabel('Inference time [ms]')
plt.ylabel('# MSOPs')

#Processing time/latency plot
plt.figure()
dvs = plt.errorbar(lat_times, Tinf_mean_per_chip[0,:]*1e3, Tinf_std_per_chip[0,:]*1e3, marker='o')
emg = plt.errorbar(lat_times, Tinf_mean_per_chip[1,:]*1e3, Tinf_std_per_chip[1,:]*1e3, marker='o')
fus = plt.errorbar(lat_times, Tinf_mean*1e3, Tinf_std*1e3, marker='o')
plt.legend([dvs,emg,fus], ["DVS data (MorphIC)", "EMG data (ODIN)", "Sensor fusion"], loc="upper left")
plt.xlabel('Inference time [ms]')
plt.ylabel('Processing time [ms]')

#Energy/latency plot
plt.figure()
dvs = plt.errorbar(lat_times, Einf_mean_per_chip[0,:]*1e6, Einf_std_per_chip[0,:]*1e6, marker='o')
emg = plt.errorbar(lat_times, Einf_mean_per_chip[1,:]*1e6, Einf_std_per_chip[1,:]*1e6, marker='o')
fus = plt.errorbar(lat_times, Einf_mean*1e6, Einf_std*1e6, marker='o')
plt.legend([dvs,emg,fus], ["DVS data (MorphIC)", "EMG data (ODIN)", "Sensor fusion"], loc="upper left")
plt.xlabel('Inference time [ms]')
plt.ylabel('Dynamic energy per classification [uJ]')

#EDP/latency plot
plt.figure()
dvs = plt.errorbar(lat_times, Pinf_mean_per_chip[0,:]*1e6, Pinf_std_per_chip[0,:]*1e6, marker='o')
emg = plt.errorbar(lat_times, Pinf_mean_per_chip[1,:]*1e6, Pinf_std_per_chip[1,:]*1e6, marker='o')
fus = plt.errorbar(lat_times, Pinf_mean*1e6, Pinf_std*1e6, marker='o')
plt.legend([dvs,emg,fus], ["DVS data (MorphIC)", "EMG data (ODIN)", "Sensor fusion"], loc="upper left")
plt.xlabel('Inference time [ms]')
plt.ylabel('EDP [uJ*s]')

plt.show()        