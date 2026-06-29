import pandas as pd
import numpy as np
import os
import glob

#RealDISP
log_path =  glob.glob('Dataset/realdisp+activity+recognition+dataset/*_ideal.log')

print(f"Found : {len(log_path)}")
print(f"Example : {log_path[0]}")

# Load only the coloumns for the lower back
BACK_COLS = [54, 55, 56, 57, 58, 59, 119]
COL_NAMES = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'label']

"""test_df = pd.read_csv(log_path[0], sep="\t", header=None, usecols=BACK_COLS)
test_df.columns = COL_NAMES

print(f"Loaded {log_path[0]}: {test_df.shape}")
print(f"Label counts:\n{test_df['label'].value_counts().sort_index()}")"""

# Window crop
def extract_windows_from_log(log_files, back_cols, col_names, window_size=128, step_size=64):
    # File Reading
    df = pd.read_csv(log_files, sep='\t', header=None, usecols=back_cols)
    df.columns = col_names

    # Numpy conversion
    data = df[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']].values
    labels = df['label'].values.astype(int)

    # Window Sliding
    windows = []
    window_labels = []

    for start in range(0, len(data)-window_size, step_size):
        end = start + window_size
        window = data[start:end]
        chunk_labels = labels[start:end]
        label = pd.Series(chunk_labels).mode()[0]

        windows.append(window)
        window_labels.append(label)
    
    # Numpy Conversion
    windows_arr = np.array(windows)
    labels_arr = np.array(window_labels)
    
    return windows_arr, labels_arr

print("="*50)
print("Processing all RealDISP log files")
print("="*50)

all_windows = []
all_labels = []

for f in sorted(log_path):
    print(f"Processing : {os.path.basename(f)}")
    w_arr, l_arr = extract_windows_from_log(f, BACK_COLS, COL_NAMES)
    all_windows.append(w_arr)
    all_labels.append(l_arr)
    print(f"Windows: {w_arr.shape[0]}")

combined_windows = np.vstack(all_windows)
combined_labels = np.concatenate(all_labels)

print(f"\nTotal combined windows: {combined_windows.shape}")
print(f"Total labels: {combined_labels.shape}")

#Clipping
combined_windows[:,:,0:3] = np.clip(combined_windows[:,:,0:3], -157.0, 157.0)
combined_windows[:,:,3:6] = np.clip(combined_windows[:,:,3:6], -34.9, 34.9)

print(f"After clip AccX max: {combined_windows[:,:,0].max():.4f}")

N = combined_windows.shape[0]
flat = combined_windows.reshape(N, -1)

cols = []
for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
    for i in range(1, 129):
        cols.append(f"{axis}_{i}")

# Labeling on the dataframe
df_out = pd.DataFrame(flat, columns=cols)
df_out['label'] = combined_labels

# Save
os.makedirs("Dataset/archive/Downsampled/Windowed", exist_ok= True)
df_out.to_csv("Dataset/archive/Downsampled/Windowed/RealDISP_50Hz_2.56s.csv", index = False)

print("Saved ", df_out.shape)
"""
#UCI HAR Dataset
base_path = 'Dataset/UCI HAR Dataset'

train_files = {
    'AccX': f'{base_path}/train/Inertial Signals/body_acc_x_train.txt',
    'AccY': f'{base_path}/train/Inertial Signals/body_acc_y_train.txt',
    'AccZ': f'{base_path}/train/Inertial Signals/body_acc_z_train.txt',
    'GyrX': f'{base_path}/train/Inertial Signals/body_gyro_x_train.txt',
    'GyrY': f'{base_path}/train/Inertial Signals/body_gyro_y_train.txt',
    'GyrZ': f'{base_path}/train/Inertial Signals/body_gyro_z_train.txt',
}
test_files = {
    'AccX': f'{base_path}/test/Inertial Signals/body_acc_x_test.txt',
    'AccY': f'{base_path}/test/Inertial Signals/body_acc_y_test.txt',
    'AccZ': f'{base_path}/test/Inertial Signals/body_acc_z_test.txt',
    'GyrX': f'{base_path}/test/Inertial Signals/body_gyro_x_test.txt',
    'GyrY': f'{base_path}/test/Inertial Signals/body_gyro_y_test.txt',
    'GyrZ': f'{base_path}/test/Inertial Signals/body_gyro_z_test.txt',
}
def load_inertial_signals(file_dict):
    data = {}
    for axis_name, filepath in file_dict.items():
        df = pd.read_csv(filepath, header=None, sep='\s+')
        data[axis_name] = df
        print(f"Loaded {axis_name}: {df.shape}")
    return data

#Loading Train
print("=" * 40)
print("Loading Train")
train_data = load_inertial_signals(train_files)

# Load Test
print("\n" + "=" *40)
print("Loading Test")
test_data = load_inertial_signals(test_files)

y_train = pd.read_csv(f'{base_path}/train/y_train.txt', header=None, names=['label'])
print(f'\nTrain labels: {y_train.shape}')

y_test = pd.read_csv(f'{base_path}/test/y_test.txt', header=None, names=['label'])
print(f'\nTest labels: {y_test.shape}')

# Train + Test
combined = {}
for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
    combined[axis] = pd.concat([train_data[axis], test_data[axis]], ignore_index=True)
    print(f"Combined {axis}: {combined[axis].shape}")

acc_x = combined['AccX'].values
acc_y = combined['AccY'].values
acc_z = combined['AccZ'].values
gyr_x = combined['GyrX'].values
gyr_y = combined['GyrY'].values
gyr_z = combined['GyrZ'].values

windows_array = np.stack([acc_x, acc_y, acc_y, gyr_x, gyr_y, gyr_z], axis = 2)

print(f"Stacked array shape: {windows_array.shape}")

labels = pd.concat([y_train,y_test], ignore_index=True)['label'].values
print('Combined labels: ',labels.shape)

#Acc
windows_array[:,:,0:3] = np.clip(windows_array[:,:,0:3], -20.0, 20.0)
#Gyro
windows_array[:,:,3:6] = np.clip(windows_array[:,:,3:6], -20.0, 20.0)
print(f"After clip AccX max: {windows_array[:,:,0].max():.4f}")

# Saving DataFrame
N = windows_array.shape[0]
flat = windows_array.reshape(N, -1)

# Naming columns
cols = []
for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
    for i in range(1, 129):
        cols.append(f"{axis}_{i}")

# Labeling on the dataframe
df_out = pd.DataFrame(flat, columns=cols)
df_out['label'] = labels

# Save
os.makedirs("Dataset/archive/Downsampled/Windowed", exist_ok= True)
df_out.to_csv("Dataset/archive/Downsampled/Windowed/UCI_HAR_50Hz_2.56s.csv", index = False)

print("Saved ", df_out.shape)

"""
"""
HARTH
# Default values for windowing
window_size = 128
step_size = 64

windows = []
window_labels = []

# load HARTH train
harth_train_path = pd.read_csv('Dataset/archive/train.csv')
# load HARTH test
harth_test_path = pd.read_csv('Dataset/archive/test.csv')

print(f"Train Shape: {harth_train_path.shape}")
print(f"Test Shape: {harth_test_path}")

harth_combine = pd.concat([harth_test_path, harth_train_path], ignore_index= True)

print(f"Combined shape: {harth_combine.shape}")
print(f"Columns: {list(harth_combine)}")

harth_combine['Activity'] = harth_combine['Activity'].astype(int)
print(f"Unique labels: {sorted(harth_combine['Activity'].unique())}")
print(f"Label counts: \n{harth_combine['Activity'].value_counts().sort_index()}")

data_6axis = harth_combine[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6']].values

# Extract labels
labels = harth_combine['Activity'].values
print("6-axis data shape: ", data_6axis.shape)
print("Labels shape: ", labels.shape)

# Windowing ---------------------------------------
for start in range(0, len(data_6axis) - window_size, step_size):
    end = start + window_size

    window = data_6axis[start:end]

    label_segment = labels[start:end]
    label = pd.Series(label_segment).mode()[0]

    windows.append(window)
    window_labels.append(label)

print("Total windows created: ", len(windows))
print("Each window shape: ", windows[0].shape)

# Covert to numpy
windows_arr = np.array(windows)
labels_arr = np.array(window_labels)

# Crop (Default)
windows_arr[:,:,0:3] = np.clip(windows_arr[:,:,0:3], -157.0, 157.0) #Acc
windows_arr[:,:,3:6] = np.clip(windows_arr[:,:,3:6], -34.9, 34.9) # Gyro
print(f"After clip AccX max: {windows_arr[:,:,0].max():.4f}")

# Creating Dataframe
N = windows_arr.shape[0]
flat = windows_arr.reshape(N, -1)

# Naming columns
cols = []
for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
    for i in range(1, 129):
        cols.append(f"{axis}_{i}")

# Labeling on the dataframe
df_out = pd.DataFrame(flat, columns=cols)
df_out['label'] = labels_arr

# Save
os.makedirs("Dataset/archive/Downsampled/Windowed", exist_ok= True)
df_out.to_csv("Dataset/archive/Downsampled/Windowed/HARTH_50Hz_2.56s.csv", index = False)

print("Saved ", df_out.shape)
"""

input_path = 'Dataset/KU-HAR/3.Time_domain_subsamples/KU-HAR_time_domain_subsamples_20750x300.csv'
df = pd.read_csv(input_path, header=None)

print(f"[INFO] Raw data shape: {df.shape}")
acc_x = df.iloc[:, 0:300].values
acc_y = df.iloc[:, 300:600].values
acc_z = df.iloc[:, 600:900].values
gyro_x = df.iloc[:, 900:1200].values
gyro_y = df.iloc[:, 1200:1500].values
gyro_z = df.iloc[:, 1500:1800].values
labels = df.iloc[:, 1800].values

print(f"[INFO] Raw AccX max: {acc_x.max():.4f}, min: {acc_x.min():.4f}")
print(f"[INFO] Raw AccX 99th percentile: {np.percentile(acc_x, 99):.4f}")
print(f"[INFO] Raw AccX 1st percentile: {np.percentile(acc_x, 1):.4f}")

ACC_LIMIT = 157.0
GYRO_LIMIT = 34.9

acc_x = np.clip(acc_x, -ACC_LIMIT, ACC_LIMIT)
acc_y = np.clip(acc_y, -ACC_LIMIT, ACC_LIMIT)
acc_z = np.clip(acc_z, -ACC_LIMIT, ACC_LIMIT)

gyro_x = np.clip(gyro_x, -GYRO_LIMIT, GYRO_LIMIT)
gyro_y = np.clip(gyro_y, -GYRO_LIMIT, GYRO_LIMIT)
gyro_z = np.clip(gyro_z, -GYRO_LIMIT, GYRO_LIMIT)

print(f"After clipping AccX max: {acc_x.max():.4f}, min: {acc_x.min():.4f}")
print(f"After clipping GyroX max: {gyro_x.max():.4f}, min: {gyro_x.min():.4f}")

acc_x_50 = acc_x[:, ::2][:, :128]
acc_y_50 = acc_y[:, ::2][:, :128]
acc_z_50 = acc_z[:, ::2][:, :128]
gyro_x_50 = gyro_x[:, ::2][:, :128]
gyro_y_50 = gyro_y[:, ::2][:, :128]
gyro_z_50 = gyro_z[:, ::2][:, :128]

print(f"Downsampled AccX max: {acc_x_50.max():.4f}, min: {acc_x_50.min():.4f}")
print(f"Downsampled GyroX max: {gyro_x_50.max():.4f}, min: {gyro_x_50.min():.4f}")


columns = []
for axis in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']:
    for i in range(1, 129):
        columns.append(f"{axis}_{i}")
columns.append('label')

data_all = np.hstack([
    acc_x_50, acc_y_50, acc_z_50,
    gyro_x_50, gyro_y_50, gyro_z_50,
    labels.reshape(-1, 1)
])

print(f" Final array shape: {data_all.shape}")

df_processed = pd.DataFrame(data_all, columns=columns)

os.makedirs("Dataset/KU-HAR/Downsampled", exist_ok=True)
output_path = 'Dataset/archive/Downsampled/Windowed/KU-HAR_50Hz_2.56s.csv'
df_processed.to_csv(output_path, index=False)

print(f" Saved to: {output_path}")

df_check = pd.read_csv(output_path)
print(f"Reloaded shape: {df_check.shape}")

acc_x_cols = [c for c in df_check.columns if c.startswith('AccX')]
print(f"Saved AccX max: {df_check[acc_x_cols].values.max():.4f}")
print(f"Saved AccX min: {df_check[acc_x_cols].values.min():.4f}")
print(f"Labels: {sorted(df_check['label'].unique())}")
