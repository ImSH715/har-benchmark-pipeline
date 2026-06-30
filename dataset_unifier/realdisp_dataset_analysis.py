import pandas as pd
import glob

# Only load the back sensor data from realdisp raw dataset
back_cols = list(range(54,67)) + [119] # 119 -> label

files  = glob.glob('Dataset/realdisp+activity+recognition+dataset/*_ideal.log')
#label
labels = pd.read_csv('Dataset/realdisp+activity+recognition+dataset/activity_labels.txt', sep='\s+', header=None, names=['number', 'name'])

print(f"{len(files)} files found in the dataset folder.")

all_data = []

for f in files:
    # Load the data
    df = pd.read_csv(f, sep='\t', header=None, usecols=back_cols)
    all_data.append(df)

# Combine all data
total = pd.concat(all_data, ignore_index=True)

print("--------- RealDisp Dataset----------")


total['label'] = total[119].astype(int)
total_name_dict = dict(zip(labels['number'], labels['name']))
total['activity_name'] = total['label'].map(total_name_dict)
print("total number of data in the each class: ", len(total))
print("\nClass distribution:", total['activity_name'].value_counts().sort_index())