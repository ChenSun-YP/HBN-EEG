import mne
import pandas as pd
import numpy as np

# print("hi")

# Load EEGLAB .set file
raw = mne.io.read_raw_eeglab(
    "src/data/raw/HBN_BIDS_EEG/cmi_bids_R1_mini/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-surroundSupp_run-2_eeg.set",
    preload=False,
)

# Number of time samples
n_annotations = raw.annotations.count
print("Number of annotations:", n_annotations)


# Plot data
raw.plot(n_channels=40, scalings=dict(eeg=1e-4))

# Get events
# events, event_id = mne.events_from_annotations(raw)

# events_df = pd.read_csv("hbn_bids_R1sample/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-surroundSupp_run-1_events.tsv", sep="\t")
# print(events_df.head())


import matplotlib.pyplot as plt

plt.subplots_adjust(top=0.95)
plt.show()

# stim_on_segments = []


# for annot in raw.annotations:
#     if annot["description"].lower() == "stim_on":
#         onset_sample = int(
#             annot["onset"] * raw.info["sfreq"]
#         )  # when the annotation begins using in the sampling index
#         # print("onset_sample" + str(onset_sample))
#         duration_samples = int(
#             annot["duration"] * raw.info["sfreq"]
#         )  # how long annotation goes in sampling index
#         # print("duration_sample" + str(duration_samples))
#         # segment = raw[
#         #     :, onset_sample : onset_sample + duration_samples
#         # ]  # [all 128 channels, start index, end index]

#         segment = raw.get_data(start=onset_sample, stop=onset_sample + duration_samples)

#         # print(segment.shape)
#         stim_on_segments.append(segment)


# # for i, seg in enumerate(stim_on_segments):
# #     print(f"{i}: {type(seg)}", end="d ")
# #     try:
# #         print("seg shape")
# #         print(seg.shape)
# #     except AttributeError:
# #         print("Attribute error - seg")
# #         print(seg)  # Not an ndarray — print raw content

# # for seg in stim_on_segments:
# #     print(type(seg), seg.shape)

# print(np.concatenate(stim_on_segments, axis=1) if stim_on_segments else None)
# print(len(stim_on_segments))
# print("done")


#
# Surround Suppression Recording
#
# HBN-EEG/src/data/raw/HBN_BIDS_EEG/cmi_bids_R1_mini/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-surroundSupp_run-2_eeg.set
# HBN-EEG/src/data/raw/HBN_BIDS_EEG/cmi_bids_R1_mini/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-surroundSupp_run-1_events.tsv

#
# CCD Recording
#
# HBN-EEG/src/data/raw/HBN_BIDS_EEG/cmi_bids_R1_mini/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-contrastChangeDetection_run-2_eeg.set
# HBN-EEG/src/data/raw/HBN_BIDS_EEG/cmi_bids_R1_mini/sub-NDARAM704GKZ/eeg/sub-NDARAM704GKZ_task-contrastChangeDetection_run-2_events.tsv
