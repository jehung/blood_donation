import csv
import collections


def read_data(file_in, features_list):
    with open(file_in, 'r') as f:
        file_reader = csv.DictReader(f)
        header = file_reader.fieldnames

        data_numerical = collections.defaultdict(dict)
        
        for row in file_reader:
            recordNum = row['ID']
            for f in features_list:
                data_numerical[recordNum][f] = row[f]

    with open("data_dict.pkl", "wb") as pick:
        pickle.dump(data_numerical, pick, protocol=pickle.HIGHEST_PROTOCOL)

    return data_numerical
    


