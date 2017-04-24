import csv
import collections
import pickle

def read_data(file_in, features_list, topickle=False):
    with open(file_in, 'rb') as f:
        file_reader = csv.DictReader(f)

        read_data = collections.defaultdict(dict)
        
        for row in file_reader:
            recordNum = row['ID']
            for f in features_list:
                read_data[recordNum][f] = row[f]

    if topickle:
        with open("data_dict.pkl", "wb") as pick:
            pickle.dump(read_data, pick, protocol=pickle.HIGHEST_PROTOCOL)

    return read_data

file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
numerical_features = ['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)',  'Months since First Donation']


ans = read_data(file_in, numerical_features)

