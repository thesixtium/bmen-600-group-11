import warnings

warnings.filterwarnings("ignore", message="Montage name 'standard_1005'")

from data_statistics import data_statistics
from download_data import download_data
from preprocess_data import preprocess_data

data = download_data(redo=False)

# processed = preprocess_data(data)

# data_statistics(data, processed)

# Make and train EEGNet ANN and benchmark

# Convert EEGNet ANN to SNN and benchmark

# Make and train RSNN model and benchmark

# Comparative metrics between models