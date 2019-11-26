# -*- coding: utf-8 -*-
"""
E2MG results summary for sensor fusion on the ODIN and MorphIC chips.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


lat_times = (10, 30, 50, 70, 100, 150, 200)  #milliseconds
E_per_SOP_MorphIC = 30e-12                   #30pJ  (incremental definition of the energy per synaptoc operation (SOP), cfr MorphIC paper)
E_per_SOP_ODIN    = 8.4e-12                  #8.4pJ (incremental definition of the energy per synaptoc operation (SOP), cfr ODIN paper)

#Load results
#   EMG_final_scores: accuracy results for each cross-validation set (3) after 200ms with a 3-bit 16-230-5 MLP on ODIN.
#   DVS_final_scores: accuracy results for each cross-validation set (3) after 200ms with 4x 1-bit 400-210-5 MLPs on MorphIC.
#   fusion_scores:    accuracy results for each cross-validation set (3) after different latencies (7) with the fusion network
#                     (i.e. retrained 3-bit 5-neuron output layer with 4x210+230=1070 inputs)
#   fusion_SOPs:      number of SOPs processed for DVS/MorphIC, EMG/ODIN, Fusion/ODIN for each cross-validation set (3) after different latencies (7).
#                     includes the dummy crossbar SOPs processed for DVS/MorphIC (only 210 out of 512 are actually used for computation)
#                     and the dummy crossbar SOPs processed for EMG/ODIN (only 230 out of 256 are actually used for computation).
#                     All SOPs for the last layer are used for computation, does not account for the energy cost of the external mapping table.
[EMG_final_scores, DVS_final_scores, fusion_scores, fusion_SOPs] = pickle.load(open('final_fusion.pkl', 'rb'))

#Extract results and write text summary:
print("=== FINAL SUMMARY ===")
print("= DVS final scores after 200ms =")
print("    After 200ms inference: "+str(np.mean(DVS_final_scores))+"% accuracy, with std dev "+str(np.std(DVS_final_scores))+"%")
print("= EMG final scores after 200ms =")
print("    After 200ms inference: "+str(np.mean(EMG_final_scores))+"% accuracy, with std dev "+str(np.std(EMG_final_scores))+"%")
print("= FUSION SCORES =")
fusion_mean = np.mean(fusion_scores, axis=0)
fusion_std = np.std(fusion_scores, axis=0)
for t in range(len(lat_times)):
    print("    After "+str(int(lat_times[t]))+"ms inference: "+str(fusion_mean[t])+"% accuracy, with std dev "+str(fusion_std[t])+"%")
print("= SOPs =")
chips = ["MorphIC", "ODIN   ", "fusion "]
SOP_mean_per_chip = np.mean(fusion_SOPs, axis=1)
SOP_std_per_chip = np.std(fusion_SOPs, axis=1)
SOP_mean = np.mean(np.sum(fusion_SOPs,axis=0), axis=0)
SOP_std = np.std(np.sum(fusion_SOPs,axis=0), axis=0)
E_per_SOP = np.asarray([E_per_SOP_MorphIC, E_per_SOP_ODIN, E_per_SOP_ODIN])
E_mean = np.mean(np.sum(fusion_SOPs*E_per_SOP[:,np.newaxis,np.newaxis], axis=0), axis=0)
E_std = np.std(np.sum(fusion_SOPs*E_per_SOP[:,np.newaxis,np.newaxis], axis=0), axis=0)
for t in range(len(lat_times)):
    print("    After "+str(int(lat_times[t]))+"ms inference: ")
    for i in range(3):
        print("        "+chips[i]+": "+str(SOP_mean_per_chip[i,t])+" SOPs with a std dev of "+str(SOP_std_per_chip[i,t])+" SOPs")
        

#Accuracy/latency plot showing standard deviations
plt.figure()
plt.errorbar(lat_times, fusion_mean, fusion_std, marker='o')
plt.xlabel('Inference time [ms]')
plt.ylabel('Inference accuracy [%]')

#SOP/latency plot
plt.figure()
plt.errorbar(lat_times, SOP_mean/1e6, SOP_std/1e6, marker='o')
plt.xlabel('Inference time [ms]')
plt.ylabel('# MSOPs')

#Energy/latency plot
plt.figure()
plt.errorbar(lat_times, E_mean*1e6, E_std*1e6, marker='o')
plt.xlabel('Inference time [ms]')
plt.ylabel('Dynamic energy per classification [uJ]')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        