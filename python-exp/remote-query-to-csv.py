from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from iotdb.utils.Tablet import Tablet

from matplotlib import pyplot as plt
import numpy as np
import csv
import datetime
import pandas as pd
import time
import argparse
import sys
import os
import math
import re
import subprocess

def myDeduplicate(seq): # deduplicate list seq by comparing the first element, e.g. l=[(1,1),(1,2)] => l=[(1,1)]
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x[0] in seen or seen_add(x[0]))]

# remote node has not exported the environment variables, so passing them using args
parser=argparse.ArgumentParser(description="remote query to csv",
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r","--read",help="READ_METHOD")
parser.add_argument("-f","--file",help="remote_M4_FILE_PATH")
parser.add_argument("-s","--tqs",help="query start time")
parser.add_argument("-e","--tqe",help="query end time")
parser.add_argument("-w","--w",help="number of time spans")
parser.add_argument("-t","--tool",help="export csv tool directory path")
args = parser.parse_args()
config = vars(args)

read_method=str(config.get('read'))
outputCsvPath=config.get('file')
print(read_method)
print(outputCsvPath)

tqs=int(config.get('tqs'))
tqe=int(config.get('tqe'))
w=int(config.get('w'))
# post-process, make divisible
interval=math.ceil((tqe-tqs)/w)
# tqe=tqs+interval*w

if read_method == 'raw':
  sql="select test from root.RTDMORE.targetDevice where time>={} and time<{}".format(tqs,tqe)
else: # lttb/ilts, actually controlled by enableTri
  sql="select min_value(test),max_value(test),min_time(test), max_time(test), first_value(test), last_value(test)\
    from root.RTDMORE.targetDevice group by ([{}, {}), {}ms)".format(tqs,tqe,interval)

print(sql)


if read_method == 'raw':
  exportCsvPath=str(config.get('tool'))+"/export-csv.sh"
  start = time.time_ns()
  os.system("bash {} -h 127.0.0.1 -p 6667 -u root -pw root -q '{}' -td {} -tf timestamp".format(exportCsvPath,sql,str(config.get('tool'))))
  end = time.time_ns()
  print(f"[2-ns]Server_Query_Execute,{end - start}") # print metric

if read_method == 'lttb' or read_method == 'ilts':
  ip = "127.0.0.1"
  port_ = "6667"
  username_ = "root"
  password_ = "root"
  fetchsize = 100000 # make it big enough to ensure no second fetch, for result.todf_noFetch
  session = Session(ip, port_, username_, password_, fetchsize)
  session.open(False)

  result = session.execute_query_statement(sql) # server execute metrics have been collected by session.execute_finish()

  start = time.time_ns() # for parse_data metric
  df = result.todf_noFetch() # Transform to Pandas Dataset
  # for each row, extract four points, sort and deduplicate, deal with empty
  ts=[]
  for ir in df.itertuples():
    string = ir[2] # ir[0] is idx
    # deal with "empty" string
    if str(string)=="empty":
      # print("empty")
      continue

    # deal with for example "5.0[1],10.0[2],2.0[40],5.0[55],20.0[62],1.0[90],7.0[102],"
    # remove global first and last points
    data_list = string.strip(",").split(",")[1:-1]
    ts = [[float(item.split("[")[1][:-1]), float(item.split("[")[0])] for item in data_list]

  # sort
  ts.sort(key=lambda x: x[0])

  # deduplicate
  ts=myDeduplicate(ts)

  df = pd.DataFrame(ts,columns=['time','value'])
  df.to_csv(outputCsvPath, sep=',',index=False)

  end = time.time_ns()
  print(f"[1-ns]parse_data,{end - start}") # print metric

  result = session.execute_finish()
  print(result) # print metrics from IoTDB server
  session.close()