import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import mode

dataset_text = pandas.read_csv('NOTEEVENTS_id_text_sorted.csv')
df = dataset_text[:10]
df.to_csv('NOTEEVENTS_see_10.csv')


########MANIPULTE THE DATA#######
dataset = pandas.read_csv('DIAGNOSES_ICD.csv')
print dataset.head()
print dataset.shape

dataset = dataset.sort('SUBJECT_ID')
df = dataset[['SUBJECT_ID','ICD9_CODE']]
df.to_csv('DIAGNOSES_ICD_id_codes_sorted.csv')



dataset = pandas.read_csv('NOTEEVENTS.csv')
print dataset.head()
print dataset.shape

dataset = dataset.sort('SUBJECT_ID')
df = dataset[['SUBJECT_ID','TEXT']]
df.to_csv('NOTEEVENTS_id_text_sorted.csv')


dataset = pandas.read_csv('DIAGNOSES_ICD_id_codes_sorted.csv')

for y in dataset.columns:
    print dataset[y].dtype

dataset['ICD9_CODE'] = dataset['ICD9_CODE'].str[0:3]
dataset = dataset.convert_objects(convert_numeric=True)
print dataset['ICD9_CODE'].dtype

numpy_id = dataset['SUBJECT_ID'].as_matrix()
np.savetxt('test_id.csv', numpy_id, delimiter=',')



yo =[]
for index, row in dataset.iterrows():
    
    if row['ICD9_CODE']>=0 and row['ICD9_CODE']<=139:
        row['ICD9_CODE'] = 1
    elif row['ICD9_CODE']>=140 and row['ICD9_CODE']<=239:
        row['ICD9_CODE'] = 2
    elif row['ICD9_CODE']>=240 and row['ICD9_CODE']<=279:
        row['ICD9_CODE'] = 3
    elif row['ICD9_CODE']>=280 and row['ICD9_CODE']<=289:
        row['ICD9_CODE'] = 4
    elif row['ICD9_CODE']>=290 and row['ICD9_CODE']<=319:
        row['ICD9_CODE'] = 5
    elif row['ICD9_CODE']>=320 and row['ICD9_CODE']<=389:
        row['ICD9_CODE'] = 6
    elif row['ICD9_CODE']>=390 and row['ICD9_CODE']<=459:
        row['ICD9_CODE'] = 7
    elif row['ICD9_CODE']>=460 and row['ICD9_CODE']<=519:
        row['ICD9_CODE'] = 8
    elif row['ICD9_CODE']>=520 and row['ICD9_CODE']<=579:
        row['ICD9_CODE'] = 9
    elif row['ICD9_CODE']>=580 and row['ICD9_CODE']<=629:
        row['ICD9_CODE'] = 10
    elif row['ICD9_CODE']>=630 and row['ICD9_CODE']<=679:
        row['ICD9_CODE'] = 11
    elif row['ICD9_CODE']>=680 and row['ICD9_CODE']<=709:
        row['ICD9_CODE'] = 12
    elif row['ICD9_CODE']>=710 and row['ICD9_CODE']<=739:
        row['ICD9_CODE'] = 13
    elif row['ICD9_CODE']>=740 and row['ICD9_CODE']<=759:
        row['ICD9_CODE'] = 14
    elif row['ICD9_CODE']>=760 and row['ICD9_CODE']<=779:
        row['ICD9_CODE'] = 15
    elif row['ICD9_CODE']>=780 and row['ICD9_CODE']<=799:
        row['ICD9_CODE'] = 16
    elif row['ICD9_CODE']>=800 and row['ICD9_CODE']<=999:
        row['ICD9_CODE'] = 17
    else:
        row['ICD9_CODE'] = 18

    yo.append(row['ICD9_CODE'])

print yo
myarray = np.asarray(yo)
np.savetxt('test.csv', myarray, delimiter=',')

array_id = np.genfromtxt("test_id.csv", delimiter=",",dtype=np.float64)
print array_id.shape
array_codes = np.genfromtxt("test_codes.csv", delimiter=",",dtype=np.float64)
print array_codes.shape

dataset = zip(array_id, array_codes)
dataset = np.asarray(dataset)

print dataset.shape
np.savetxt('test_id_codes.csv', dataset, delimiter=',')



dataset = np.genfromtxt("test_id_codes.csv", delimiter=",",dtype=np.float64)

mapping = []
lt = []
for i in range(len(dataset)-1):
    
    if i == len(dataset):
        break
    if dataset[i,0] == dataset[i+1,0]:
        lt.append(dataset[i,1])
    if dataset[i,0] != dataset[i+1,0]:
        lt.append(dataset[i,1])
        most_frequent = mode(lt)[0][0]
        mapping.append((dataset[i,0],most_frequent))
        lt = []

from collections import Counter
input =  [ seq[1] for seq in mapping ]
c = Counter( input )

print( c.items() )

dataset_text = pandas.read_csv('NOTEEVENTS.csv')

#dataset_text = dataset_text.convert_objects(convert_numeric=True)
print dataset_text['TEXT'].dtype
print dataset_text['SUBJECT_ID'].dtype
dataset_text['SUBJECT_ID'] = dataset_text['SUBJECT_ID'].apply(pandas.to_numeric)

dataset_text = dataset_text.iloc[np.random.permutation(len(dataset_text))]
dataset_text = dataset_text.reset_index(drop=True)
dataset_text_train = dataset_text[:11000]

#df = dataset[['SUBJECT_ID','TEXT']]
#dataset_text['FINAL CODES'] = np.nan

print "Preparing data for final use"

lolo = []
for index, row in dataset_text_train.iterrows():
    print index
    for i in range(len(mapping)):
        
        if row['SUBJECT_ID'] == mapping[i][0]:
            #print mapping[i][0]
            #print row['SUBJECT_ID']
            get_label = mapping[i][1]
            
            break
        else:
            get_label = 0
            #print mapping[i][1]
            #print row['FINAL CODES']
    lolo.append(get_label)
        

print lolo
df1 = pandas.DataFrame({'FINAL CODES': lolo})
dataset_text_train['FINAL CODES'] = df1.values
#df = dataset_text[['SUBJECT_ID','TEXT','FINAL CODES']]
dataset_text_train.to_csv('data_11000.csv')      


print "Preparing data for testing"

dataset_text_test = dataset_text[5000:5500]
lolo = []
for index, row in dataset_text_test.iterrows():
    print index
    for i in range(len(mapping)):
        
        if row['SUBJECT_ID'] == mapping[i][0]:
            #print mapping[i][0]
            #print row['SUBJECT_ID']
            get_label = mapping[i][1]
            
            break
        else:
            get_label = 0
    lolo.append(get_label)
            #print mapping[i][1]
            #print row['FINAL CODES']
            



print lolo
df2 = pandas.DataFrame({'FINAL CODES': lolo})
#.loc[row_indexer,col_indexer] = value instead
dataset_text_test['FINAL CODES'] = df2.values
#df = dataset_text[['SUBJECT_ID','TEXT','FINAL CODES']]
dataset_text_test.to_csv('data_test_5000_5500.csv')   


