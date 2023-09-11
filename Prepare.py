import pandas as pd


df = pd.read_csv('./soybean_data_soilgrid250_modified_states_9.csv')

for index1 in range(2019-30,2019):
    df1 = df.loc[df['year']==index1]
    count=0
    for index,row in df1.iterrows():
        while count != row['loc_ID'] and count <= row['loc_ID']:
            df.drop(df[df['loc_ID'] ==count].index, inplace=True)
            count = count+1
        else:
            count = count +1
    while count<=1045:
        df.drop(df[df['loc_ID'] ==count].index, inplace=True)
        count = count+1
        
num_exist1 = df1['loc_ID'].value_counts()
for index,row in df.iterrows():
    if row['year']<2019-30:
        df.drop([index],inplace = True)
print(df.corr())
df.to_csv('Pre.csv', index = True)