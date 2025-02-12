#!/bin/bash

# generate HOME_PATH workspace by running prepare-all.sh first
HOME_PATH=/data/v4
TRI_VISUALIZATION_EXP=/root/lts-exp

# check if experiments.jar prepared
if [ -f "experiments.jar" ]; then
    echo "experiments.jar exists."
else
    echo "Error: experiments.jar does not exist. Please mvn clean package from MinMaxCache repository \
    to create experiments.jar and copy it here!" >&2
    exit 1
fi

# dataset basic info
DATASET=BallSpeed
DEVICE="root.game"
MEASUREMENT="s6"
DATA_TYPE=long
TIMESTAMP_PRECISION=ns
DATA_MIN_TIME=0  # in the corresponding timestamp precision
DATA_MAX_TIME=617426057626  # in the corresponding timestamp precision
TOTAL_POINT_NUMBER=1200000
let TOTAL_TIME_RANGE=${DATA_MAX_TIME}-${DATA_MIN_TIME}
VALUE_ENCODING=PLAIN
TIME_ENCODING=PLAIN
COMPRESSOR=UNCOMPRESSED
INPUT_DATA_PATH=$HOME_PATH/${DATASET}/${DATASET}.csv

IOTDB_CLIENT_TOOL_DIR=$HOME_PATH/iotdb-cli-0.12.4/tools

output="output${DATASET}"
accuracy=0.95
querySelectivity=0.1
prefetching=1
aggFactorMinMax=4
# because ILTS samples one point per bucket while minmax samples two
aggFactorILTS=$((aggFactorMinMax * 2))
echo $aggFactorILTS

realT=false

width=1000
height=250

# iotdb config info
IOTDB_CHUNK_POINT_SIZE=1000

FIX_QUERY_RANGE=$TOTAL_TIME_RANGE

#hasHeader=false # default

otherSeriesCnt=4

seqCount=50

echo 3 |sudo tee /proc/sys/vm/drop_cache
free -m
echo "Begin experiment!"


cd $HOME_PATH/${DATASET}_testspace
mkdir O_10_D_0_0
cd O_10_D_0_0

# prepare IoTDB config properties
$HOME_PATH/tool.sh system_dir $HOME_PATH/dataSpace/${DATASET}_O_10_D_0_0/system ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh data_dirs $HOME_PATH/dataSpace/${DATASET}_O_10_D_0_0/data ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh wal_dir $HOME_PATH/dataSpace/${DATASET}_O_10_D_0_0/wal ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh timestamp_precision ${TIMESTAMP_PRECISION} ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh unseq_tsfile_size 1073741824 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh seq_tsfile_size 1073741824 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh avg_series_point_number_threshold ${IOTDB_CHUNK_POINT_SIZE} ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh compaction_strategy NO_COMPACTION ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh enable_unseq_compaction false ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh group_size_in_byte 1073741824 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh page_size_in_byte 1073741824 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh wal_buffer_size 1073741824 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh rpc_address 0.0.0.0 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh rpc_port 6667 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh time_encoder ${TIME_ENCODING} ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh compressor ${COMPRESSOR} ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh error_Param 50 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh meta_data_cache_enable false ../../iotdb-engine-example.properties # note this!
$HOME_PATH/tool.sh write_convex_hull true ../../iotdb-engine-example.properties # note this!
$HOME_PATH/tool.sh auto_p1n true ../../iotdb-engine-example.properties # note this!
$HOME_PATH/tool.sh enable_Tri ILTS ../../iotdb-engine-example.properties # note this!
$HOME_PATH/tool.sh numIterations 4 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh acc_avg true ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh acc_rectangle true ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh acc_convex true ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh acc_iterRepeat true ../../iotdb-engine-example.properties

# note
# enlarge memory allocation for write when writing
$HOME_PATH/tool.sh write_read_schema_free_memory_proportion 6:1:1:2 ../../iotdb-engine-example.properties
$HOME_PATH/tool.sh MAX_HEAP_SIZE \"5G\" $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-env.sh

cp ../../iotdb-engine-example.properties iotdb-engine-USE.properties

#######################################################
# [write data]
# if already written, this will be omitted automatically
echo "Writing data $DATASET"
cp iotdb-engine-USE.properties $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
cd $HOME_PATH/iotdb-server-0.12.4/sbin
./start-server.sh /dev/null 2>&1 &
sleep 8s
start_time=$(date +%s%N)
# Usage: java -jar WriteDataMore*.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath otherSeriesCnt
java -jar $HOME_PATH/WriteDataMore*.jar ${DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_TYPE} ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${INPUT_DATA_PATH} ${otherSeriesCnt}
end_time=$(date +%s%N)
duration_ns=$((end_time - start_time))
echo "write latency of $DATASET (with convex hull) is: $duration_ns ns"
sleep 5s
./stop-server.sh
sleep 5s
echo 3 | sudo tee /proc/sys/vm/drop_caches

#######################################################
echo "Querying data by MinMaxCache"
echo 3 | sudo tee /proc/sys/vm/drop_caches
cd $HOME_PATH/iotdb-server-0.12.4/sbin
./start-server.sh /dev/null 2>&1 &
sleep 8s

cd $HOME_PATH
java -jar $HOME_PATH/experiments.jar \
  -c timeQueries \
  -seqCount ${seqCount} \
  -schema ${DEVICE} \
  -table ${MEASUREMENT} \
  -sampleInterval 1 \
  -fromRange ${DATA_MIN_TIME} \
  -toRange ${DATA_MAX_TIME} \
  -measureChange 0 \
  -type iotdb \
  -mode minMax \
  -measures 0 \
  -zoomFactor 2 \
  -viewport ${width},${height} \
  -runs 1 \
  -out ${output} \
  -minShift 0.1 \
  -maxShift 0.5 \
  -a ${accuracy} \
  -q ${querySelectivity} \
  -p ${prefetching} \
  -agg ${aggFactorMinMax} \
  -realT ${realT}

cd $HOME_PATH/iotdb-server-0.12.4/sbin
./stop-server.sh
sleep 5s
echo 3 | sudo tee /proc/sys/vm/drop_caches

#######################################################
echo "Querying data by ILTSCache"
echo 3 | sudo tee /proc/sys/vm/drop_caches
cd $HOME_PATH/iotdb-server-0.12.4/sbin
./start-server.sh /dev/null 2>&1 &
sleep 8s

cd $HOME_PATH
java -jar experiments.jar \
  -c timeQueries \
  -seqCount ${seqCount} \
  -schema ${DEVICE} \
  -table ${MEASUREMENT} \
  -sampleInterval 1 \
  -fromRange ${DATA_MIN_TIME} \
  -toRange ${DATA_MAX_TIME} \
  -measureChange 0 \
  -type iotdb \
  -mode ilts \
  -measures 0 \
  -zoomFactor 2 \
  -viewport ${width},${height} \
  -runs 1 \
  -out ${output} \
  -minShift 0.1 \
  -maxShift 0.5 \
  -q ${querySelectivity} \
  -p ${prefetching} \
  -agg ${aggFactorILTS} \
  -realT ${realT}

cd $HOME_PATH/iotdb-server-0.12.4/sbin
./stop-server.sh
sleep 5s
echo 3 | sudo tee /proc/sys/vm/drop_caches

#######################################################
echo "Exporting raw data"
echo 3 | sudo tee /proc/sys/vm/drop_caches
cd $HOME_PATH/iotdb-server-0.12.4/sbin
./start-server.sh /dev/null 2>&1 &
sleep 8s

cd $HOME_PATH
mkdir dump_${DATASET}
cd $HOME_PATH/dump_${DATASET}
bash ${IOTDB_CLIENT_TOOL_DIR}/export-csv.sh -h 127.0.0.1 -p 6667 -u root -pw root \
  -s $HOME_PATH/${output}/timeQueries/iotdb/${DEVICE}.${MEASUREMENT}/run_0/iltsResults/rawQuery.csv \
  -td . \
  -tf timestamp

cd $HOME_PATH/iotdb-server-0.12.4/sbin
./stop-server.sh
sleep 5s
echo 3 | sudo tee /proc/sys/vm/drop_caches

#######################################################
echo "compare and plot results"
cd ${TRI_VISUALIZATION_EXP}/notebook

python3 compare-cache-exp.py \
  --width ${width} \
  --height ${height} \
  --seqCount ${seqCount} \
  --timeseries ${DEVICE}.${MEASUREMENT} \
  --raw_template $HOME_PATH/dump_${DATASET}/dump{}.csv \
  --output $HOME_PATH/${output}

cd $HOME_PATH
cp ${TRI_VISUALIZATION_EXP}/notebook/exp-minmaxcache-interaction-new.png $HOME_PATH/exp-cache.png
cp ${TRI_VISUALIZATION_EXP}/notebook/exp-minmaxcache-interaction-new.eps $HOME_PATH/exp-cache.eps

cp $HOME_PATH/${output}/cache-ssim.csv $HOME_PATH/exp-cache-ssim.csv
cp $HOME_PATH/${output}/timeQueries/iotdb/${DEVICE}.${MEASUREMENT}/run_0/iltsResults/results.csv \
  $HOME_PATH/exp-cache-ilts-query.csv
cp $HOME_PATH/${output}/timeQueries/iotdb/${DEVICE}.${MEASUREMENT}/run_0/minMaxResults/results.csv \
  $HOME_PATH/exp-cache-minMax-query.csv

#######################################################
echo "ALL FINISHED!"
echo "plotted results in exp-cache.png"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m