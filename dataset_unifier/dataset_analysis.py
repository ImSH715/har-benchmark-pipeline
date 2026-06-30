import pandas as pd

# UCI HAR Dataset
UCI_HAR_train = pd.read_csv('Dataset/UCI HAR Dataset/train/y_train.txt', header=None, names=['label'])
UCI_HAR_test = pd.read_csv('Dataset/UCI HAR Dataset/test/y_test.txt', header=None, names=['label'])
# Test + Train
UCI_HAR_total = pd.concat([UCI_HAR_train, UCI_HAR_test], ignore_index=True)
UCI_HAR_label = pd.read_csv('Dataset/UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, names=['number', 'name'])

# HARTH Dataset
HART_train = pd.read_csv('Dataset/archive/train.csv')
HART_label = pd.read_csv('Dataset/archive/activity_labels.txt', sep='\s+', header=None, names=['number', 'name'])
# Total
HART_total = pd.concat([HART_train], ignore_index=True)

# HAR70+ Dataset
HAR70_total = ['501.csv', '502.csv', '503.csv', '504.csv', '505.csv', '506.csv', '507.csv', '508.csv', '509.csv', '510.csv', '511.csv', '512.csv', '513.csv', '514.csv', '515.csv']
HAR70_all_data = {}
HAR70_label = pd.read_csv('Dataset/HAR70/activity_labels.txt', sep='\s+', header=None, names=['number', 'name'])

# -------- UCI HAR Dataset
print("--------- UCI HAR Dataset----------")
#create dictionary
UCI_HAR_name_dict = dict(zip(UCI_HAR_label['number'], UCI_HAR_label['name']))

# Show dictionary combined label to see number of data in the each class
UCI_HAR_total['activity_name'] = UCI_HAR_total['label'].map(UCI_HAR_name_dict)
print("total number of data in the each class: ", len(UCI_HAR_total))
print(UCI_HAR_total['activity_name'].value_counts())

# -------- HART Dataset
HART_total['Activity'] = HART_total['Activity'].astype(int)
# create dictionary
HART_name_dict = dict(zip(HART_label['number'], HART_label['name']))

# Show dictionary combined label to see number of data in the each class
HART_total['activity_name'] = HART_total['Activity'].map(HART_name_dict)
print("--------- HART Dataset----------")
print("total number of data in the each class: ", len(HART_total))
print(HART_total['activity_name'].value_counts())

# -------- HAR70+ Dataset
for name in HAR70_total:
    # CSV files
    csv_file = pd.read_csv(f'Dataset/HAR70/{name}')
    # Combine all data into a dictionary
    HAR70_all_data[name] = csv_file

    print(f'{name}: Total {len(csv_file)} data points')
HAR70_combined = pd.concat(HAR70_all_data.values(), ignore_index=True)
print("--------- HAR70+ Dataset----------")
HAR70_combined['label'] = HAR70_combined['label'].astype(int)
HAR70_name_dict = dict(zip(HAR70_label['number'], HAR70_label['name']))
HAR70_combined['activity_name'] = HAR70_combined['label'].map(HAR70_name_dict)

print("total number of data in the each class: ", len(HAR70_combined))
print(HAR70_combined['activity_name'].value_counts())