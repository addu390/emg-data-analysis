import pandas as pd
from signal_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation

# Load data from Excel file
signal_path = 'data/emg.xlsx'
emg_signal = pd.read_excel(signal_path).values
channel_name = 'Right Hand'

# Sampling Frequency of 2000 (2000 Samples per second)
sampling_frequency = 2e3
frame = 500
step = 250

# Plot raw sEMG signal
plot_signal(emg_signal, sampling_frequency, channel_name)

emg_signal = emg_signal.reshape((emg_signal.size,))
# Band Stop Filter (BSF)
filtered_signal = notch_filter(emg_signal, sampling_frequency,
                               True)
# Band Pass Filter (BPF)
filtered_signal = bp_filter(filtered_signal, 10, 500,
                            sampling_frequency, True)

# EMG Feature Extraction
emg_features, features_names = features_estimation(filtered_signal, channel_name,
                                                   sampling_frequency, frame, step)
