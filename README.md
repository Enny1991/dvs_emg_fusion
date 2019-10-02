# dvs_emg_fusion

DATASET available at the link [link](https://zenodo.org/record/3228846#.XZOdjuczYWo)

Dataset structure:

EMG and DVS recordings
- 10 subjects
- 3 sessions for each subject
- 5 gestures in each session ('pinky', 'elle', 'yo', 'index', 'thumb')

Data names:

subjectXX_sessionYY_ZZZ
- XX : [01, 02, 03, 04, 05, 06, 07, 08, 09, 10] 
- YY : [01, 02, 03]
- ZZZ : [emg, ann, dvs, davis]

Data format:

- emg: .npy
- ann: .npy
- dvs: .aedat,.npy
- davis: .aedat,.mat,.npz

**DVS**

DVS recordings only contain DVS events
- .aedat (raw data): can be imported in Matlab using [this_script](https://github.com/inivation/AedatTools/tree/master/Matlab) or in Python with function aedat2numpy in [converter.py](https://github.com/Enny1991/hand_gestures_cc19/tree/master/jAER_utils).
- .npy (exported data): numpy.ndarray [xpos, ypos, ts, pol], 2D numpy array containing data of all events, timestamps ts reset to the trigger event (synchronized with the myo), timestamps ts in seconds.

 
**DAVIS**

DAVIS recordings contain DVS events and APS frames.
- .aedat (raw data): can be imported in Matlab using (https://github.com/inivation/AedatTools/tree/master/Matlab) or in Python with function DAVISaedat2numpy in converter.py (https://github.com/Enny1991/hand_gestures_cc19/tree/master/jAER_utils).
- .mat (exported data): mat structure, name 'aedat', events are inside aedat.data.polarity [aedat.data.polarity.x,aedat.data.polarity.y,aedat.data.polarity.timeStamp,aedat.data.polarity.polarity], aps frames are inside aedat.data.frame.samples, timestamps are in aedat.data.frame.timeStampStart (start of frame collection) or aedat.data.frame.timeStampEnd (end of frame collection)
- .npz (exported data): npz files: ['frames_time', 'dvs_events', 'frames'], 'dvs_events' is a numpy.ndarray [xpos, ypos, ts, pol], 2D numpy array containing data of all events, timestamps ts reset to the trigger event (synchronized with the myo), timestamps ts in seconds; 'frames' and 'frames_time' are aps data, 'frames' is a list of all the frames, reset at the trigger time, 'frames_time' is the time for each frame, we considered the start timeStamps for each frame.
