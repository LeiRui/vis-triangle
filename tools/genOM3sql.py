from myfuncs import *

datasetNameList=["qloss","pyra1","windspeed","rtd"]
mList=[320,480,740,1200,2000,3500,6000,10000,15000]

for datasetName in datasetNameList:
    with open(f'om3-{datasetName}.sql', 'w') as file:
        for m in mList:
            f='ids-{}-{}.csv'.format(datasetName,m)
            df = pd.read_csv(f,header=None)
            ids = df[0].tolist()
            numbers_str = ','.join(map(str, ids))
            sql_query = f"\\copy (select i, minvd, maxvd from om3.{datasetName}100_om3_16m where i in ({numbers_str}) order by i asc) TO 'om3-{datasetName}-{m}.csv' WITH CSV HEADER"
            file.write(sql_query+ '\n\n')
