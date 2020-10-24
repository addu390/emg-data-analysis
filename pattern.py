import pandas as pd
from signal_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation

# Load data
signal_path = 'data/emg.xlsx'
emg_signal = pd.read_excel(signal_path).values
channel_name = 'Right Hand'
sampling_frequency = 2e3
frame = 500
step = 250

# Plot raw sEMG signal
plot_signal(emg_signal, sampling_frequency, channel_name)

# Biomedical Signal Processing
emg_signal = emg_signal.reshape((emg_signal.size,))
filtered_signal = notch_filter(emg_signal, sampling_frequency,
                               True)
filtered_signal = bp_filter(filtered_signal, 10, 500,
                            sampling_frequency, True)

# EMG Feature Extraction
emg_features, features_names = features_estimation(filtered_signal, channel_name,
                                                   sampling_frequency, frame, step)
