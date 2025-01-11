#!/bin/bash

# generate HOME_PATH workspace by running prepare-all.sh first
HOME_PATH=/data/v4

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

# iotdb config info
IOTDB_CHUNK_POINT_SIZE=10000

FIX_QUERY_RANGE=$TOTAL_TIME_RANGE

MAX_POINTS_WRITE=-1

#hasHeader=false # default

otherSeriesCnt=0

echo 3 |sudo tee /proc/sys/vm/drop_cache
free -m
echo "Begin experiment!"

perlist="20 40 60 80 100"


for per in $perlist
do
  MAX_POINTS_WRITE=$((echo scale=0 ; echo ${per}*${TOTAL_POINT_NUMBER}/100) | bc )
  echo "per=${per}% of ${TOTAL_POINT_NUMBER}, points=${MAX_POINTS_WRITE}"

  cd $HOME_PATH/${DATASET}_testspace
  mkdir O_10_D_0_0
  cd O_10_D_0_0

  # prepare IoTDB config properties
  $HOME_PATH/tool.sh system_dir $HOME_PATH/dataSpace_noConvexHull_${per}/${DATASET}_O_10_D_0_0/system ../../iotdb-engine-example.properties # note this
  $HOME_PATH/tool.sh data_dirs $HOME_PATH/dataSpace_noConvexHull_${per}/${DATASET}_O_10_D_0_0/data ../../iotdb-engine-example.properties # note this
  $HOME_PATH/tool.sh wal_dir $HOME_PATH/dataSpace_noConvexHull_${per}/${DATASET}_O_10_D_0_0/wal ../../iotdb-engine-example.properties # note this
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
  $HOME_PATH/tool.sh meta_data_cache_enable false ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh auto_p1n false ../../iotdb-engine-example.properties

  $HOME_PATH/tool.sh write_convex_hull false ../../iotdb-engine-example.properties # note this!

  $HOME_PATH/tool.sh numIterations 4 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_avg true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_rectangle true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_convex true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_iterRepeat true ../../iotdb-engine-example.properties

  # note
  # enlarge memory allocation for write when writing
  $HOME_PATH/tool.sh write_read_schema_free_memory_proportion 6:1:1:2 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh MAX_HEAP_SIZE \"12G\" $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-env.sh

  cp ../../iotdb-engine-example.properties iotdb-engine-noCH.properties

  # [write data]
  # if already written, this will be omitted automatically
  echo "Writing data $DATASET (without convex hull)"
  cp iotdb-engine-noCH.properties $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
  cd $HOME_PATH/iotdb-server-0.12.4/sbin
  ./start-server.sh /dev/null 2>&1 &
  sleep 8s
  start_time=$(date +%s%N)
  # Usage: java -jar WriteDataMore*.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath otherSeriesCnt
  java -jar $HOME_PATH/WriteDataMore*.jar ${DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_TYPE} ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${INPUT_DATA_PATH} ${otherSeriesCnt} ${MAX_POINTS_WRITE}
  end_time=$(date +%s%N)
  duration_ns=$((end_time - start_time))
  echo "write latency of $DATASET (without convex hull) for $per is: $duration_ns ns"
  sleep 5s
  ./stop-server.sh
  sleep 5s
  echo 3 | sudo tee /proc/sys/vm/drop_caches

  ######################################################################

  # prepare IoTDB config properties
  $HOME_PATH/tool.sh system_dir $HOME_PATH/dataSpace_ConvexHull_${per}/${DATASET}_O_10_D_0_0/system ../../iotdb-engine-example.properties # note this
  $HOME_PATH/tool.sh data_dirs $HOME_PATH/dataSpace_ConvexHull_${per}/${DATASET}_O_10_D_0_0/data ../../iotdb-engine-example.properties # note this
  $HOME_PATH/tool.sh wal_dir $HOME_PATH/dataSpace_ConvexHull_${per}/${DATASET}_O_10_D_0_0/wal ../../iotdb-engine-example.properties # note this
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
  $HOME_PATH/tool.sh meta_data_cache_enable false ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh auto_p1n false ../../iotdb-engine-example.properties

  $HOME_PATH/tool.sh write_convex_hull true ../../iotdb-engine-example.properties # note this!

  $HOME_PATH/tool.sh numIterations 4 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_avg true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_rectangle true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_convex true ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh acc_iterRepeat true ../../iotdb-engine-example.properties

  # note
  # enlarge memory allocation for write when writing
  $HOME_PATH/tool.sh write_read_schema_free_memory_proportion 6:1:1:2 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh MAX_HEAP_SIZE \"12G\" $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-env.sh

  cp ../../iotdb-engine-example.properties iotdb-engine-CH.properties

  # [write data]
  # if already written, this will be omitted automatically
  echo "Writing data $DATASET (with convex hull)"
  cp iotdb-engine-CH.properties $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
  cd $HOME_PATH/iotdb-server-0.12.4/sbin
  ./start-server.sh /dev/null 2>&1 &
  sleep 8s
  start_time=$(date +%s%N)
  # Usage: java -jar WriteDataMore*.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath otherSeriesCnt
  java -jar $HOME_PATH/WriteDataMore*.jar ${DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_TYPE} ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${INPUT_DATA_PATH} ${otherSeriesCnt} ${MAX_POINTS_WRITE}
  end_time=$(date +%s%N)
  duration_ns=$((end_time - start_time))
  echo "write latency of $DATASET (with convex hull) for $per is: $duration_ns ns"
  sleep 5s
  ./stop-server.sh
  sleep 5s
  echo 3 | sudo tee /proc/sys/vm/drop_caches

done

echo "ALL FINISHED!"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m