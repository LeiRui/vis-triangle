from myfuncs import *

m="rtd-320" 
log="logs-varyM/log-{}".format(m)

with open(log, 'r') as file:
    log_text = file.read()

match = re.search(r'SQL Parameters=(\d+)', log_text)
if not match:
    raise ValueError("cannot match SQL Parameters")
    
number = int(match.group(1))  
numbers = [-1] + list(range(1, number))
    
query_pattern = r"(select .*? where  i in \((.*?)\) order by i asc)"
queries = re.findall(query_pattern, log_text)

# query_data = [(full_query, len(where_values.split(','))*2) for full_query, where_values in queries] # multiply by 2 because minmax
# print(len(query_data))

ids=numbers
for full_query, where_values in queries:
    ids.extend([int(x) for x in where_values.split(',')])
    
print(len(ids),' rows, ', len(ids)*2,' points')

df = pd.DataFrame(ids, columns=['Column1'])
df.to_csv('ids-{}.csv'.format(m), index=False, header=False)