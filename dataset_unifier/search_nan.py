import pandas as pd
import numpy as np
import glob

"""UCI HAR Dataset"""
# Acceleration
acc_x = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', header=None)
acc_y = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt', header=None)
acc_z = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt', header=None)

# Gyroscope
gyro_x = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt', header=None)
gyro_y = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt', header=None)
gyro_z = pd.read_csv('Dataset/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt', header=None)

print("UCI HAR ACC X NaN", acc_x.isnull().sum().sum())
print("UCI HAR ACC Y NaN", acc_y.isnull().sum().sum())
print("UCI HAR ACC Z NaN", acc_z.isnull().sum().sum())
print("UCI HAR GYRO X NaN", gyro_x.isnull().sum().sum())
print("UCI HAR GYRO Y NaN", gyro_y.isnull().sum().sum())
print("UCI HAR GYRO Z NaN", gyro_z.isnull().sum().sum())


"""HART Dataset"""
hart_train = pd.read_csv('Dataset/archive/train.csv', sep='\s+', header=None)

print("--------- UCI HAR Dataset----------")    

print("UCI HAR Dataset NaN", hart_train.isnull().sum().sum())

"""HAR70+ Dataset"""
datasets = ['501.csv', '502.csv', '503.csv', '504.csv', '505.csv', '506.csv', '507.csv', '508.csv', '509.csv', '510.csv', '511.csv', '512.csv', '513.csv', '514.csv', '515.csv']
HAR70_all_data = {}
for name in datasets:
    # CSV files
    csv_file = pd.read_csv(f'Dataset/HAR70/{name}')
    HAR70_all_data[name] = csv_file
    
HAR70_combined = pd.concat(HAR70_all_data.values(), ignore_index=True)
print("--------- HAR70+ Dataset----------")
print("HAR70+ Dataset NaN", HAR70_combined.isnull().sum().sum())

"""RealDisp Dataset"""
files  = glob.glob('Dataset/realdisp+activity+recognition+dataset/*_ideal.log')
# label
labels = pd.read_csv('Dataset/realdisp+activity+recognition+dataset/activity_labels.txt', sep='\s+', header=None, names=['number', 'name'])
print("--------- RealDisp Dataset----------")
print("RealDisp Dataset NaN", sum(pd.read_csv(f).isnull().sum().sum() for f in files))

"""KU-HAR Dataset"""
ku_har = pd.read_csv('Dataset/KU-HAR/3.Time_domain_subsamples/KU-HAR_time_domain_subsamples_20750x300.csv', header=None)
print("--------- KU-HAR Dataset----------")
print("KU-HAR Dataset NaN", ku_har.isnull().sum().sum())