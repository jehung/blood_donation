import csv
import pandas as pd
from discover import model, final_model
from read_data import read_data
from feature_format import featureFormat, targetFeatureSplit



file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
numerical_features = ['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)',  'Months since First Donation']

df = pd.read_csv(file_in, sep=',',header=0)
df_id = df.loc[:, ['ID']]
features = df[numerical_features]


'''
id = pd.read_csv(file_in, usecols=['ID'])

data_raw = read_data(file_in, numerical_features)
print 'HERE', len(data_raw)
data = featureFormat(data_raw, features=numerical_features)

labels, features = targetFeatureSplit(data)
'''


clf = final_model()

predict_proba = clf.predict_proba(features)
print predict_proba.shape
#print clf.classes_



test_col = pd.DataFrame(predict_proba)

test = pd.concat([df_id, test_col], axis=1)
print test
# now remove prob. for class 0

submission = test.loc[:,['ID', 1]]
submission.rename(columns={'ID': '', 1: 'Made Donation in March 2007'}, inplace=True)

submission.to_csv('submission.csv')

