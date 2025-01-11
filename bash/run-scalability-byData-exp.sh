#!/bin/bash

# generate HOME_PATH workspace by running prepare-all.sh first
HOME_PATH=/data/v4
TRI_VISUALIZATION_EXP=/root/lts-exp

# dataset basic info
DATASET=BallSpeed
DEVICE="root.game"
OM3_MIN_DEVICE="root.gameMin"
OM3_MAX_DEVICE="root.gameMax"
MEASUREMENT="s6"
DATA_TYPE=long # long or double
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
IOTDB_CHUNK_POINT_SIZE=300

#FIX_QUERY_RANGE=$TOTAL_TIME_RANGE
FIX_M=8044

#hasHeader=false # default

perlist="10 20 30 40 50 60 70 80 90 100"

otherSeriesCnt=5

echo 3 |sudo tee /proc/sys/vm/drop_cache
free -m
echo "Begin experiment!"

for per in $perlist
do
  MAX_POINTS_WRITE=$((echo scale=0 ; echo ${per}*${TOTAL_POINT_NUMBER}/100) | bc )
  echo "per=${per}% of ${TOTAL_POINT_NUMBER}, points=${MAX_POINTS_WRITE}"

  DATA_MAX_TIME=${MAX_POINTS_WRITE} # default timestamp start from 0 increment by 1
  let TOTAL_TIME_RANGE=${DATA_MAX_TIME}-${DATA_MIN_TIME}
  FIX_QUERY_RANGE=$TOTAL_TIME_RANGE
  echo "${DATA_MIN_TIME},${DATA_MAX_TIME},${FIX_QUERY_RANGE}"

  cd $HOME_PATH/${DATASET}_testspace
  mkdir O_10_D_0_0
  cd O_10_D_0_0

  # prepare IoTDB config properties
  $HOME_PATH/tool.sh system_dir $HOME_PATH/dataSpace_${per}/${DATASET}_O_10_D_0_0/system ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh data_dirs $HOME_PATH/dataSpace_${per}/${DATASET}_O_10_D_0_0/data ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh wal_dir $HOME_PATH/dataSpace_${per}/${DATASET}_O_10_D_0_0/wal ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh timestamp_precision ${TIMESTAMP_PRECISION} ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh unseq_tsfile_size 1073741824 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh seq_tsfile_size 1073741824 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh avg_series_point_number_threshold ${IOTDB_CHUNK_POINT_SIZE} ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh compaction_strategy NO_COMPACTION ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh enable_unseq_compaction false ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh group_size_in_byte 1073741824 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh page_size_in_byte 1073741824 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh rpc_address 0.0.0.0 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh rpc_port 6667 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh time_encoder ${TIME_ENCODING} ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh compressor ${COMPRESSOR} ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh error_Param 50 ../../iotdb-engine-example.properties
  $HOME_PATH/tool.sh meta_data_cache_enable false ../../iotdb-engine-example.properties # note this!
  $HOME_PATH/tool.sh write_convex_hull true ../../iotdb-engine-example.properties # note this!
  $HOME_PATH/tool.sh auto_p1n false ../../iotdb-engine-example.properties
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

  # [write data]
  # if already written, this will be omitted automatically
  echo "Writing data"
  cp iotdb-engine-USE.properties $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
  cd $HOME_PATH/iotdb-server-0.12.4/sbin
  ./start-server.sh /dev/null 2>&1 &
  sleep 8s
  start_time=$(date +%s%N)

  # Usage: java -jar WriteDataMore*.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath otherSeriesCnt
  java -jar $HOME_PATH/WriteDataMore*.jar ${DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} \
    ${DATA_TYPE} ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${INPUT_DATA_PATH} ${otherSeriesCnt} ${MAX_POINTS_WRITE}
#
#  # write om3 table two columns: minvd
#  java -jar $HOME_PATH/WriteDataOM3*.jar ${OM3_MIN_DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_TYPE} \
#    ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${TRI_VISUALIZATION_EXP}/datasets/${DATASET,,}${per}_om3_*.csv ${otherSeriesCnt} -1 1
#  # write om3 table two columns: maxvd
#  java -jar $HOME_PATH/WriteDataOM3*.jar ${OM3_MAX_DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_TYPE} \
#    ${VALUE_ENCODING} ${IOTDB_CHUNK_POINT_SIZE} ${TRI_VISUALIZATION_EXP}/datasets/${DATASET,,}${per}_om3_*.csv ${otherSeriesCnt} -1 2

  end_time=$(date +%s%N)
  duration_ns=$((end_time - start_time))
  echo "write latency of $DATASET (with convex hull) is: $duration_ns ns"
  sleep 5s
  ./stop-server.sh
  sleep 5s
  echo 3 | sudo tee /proc/sys/vm/drop_caches


  # [query data]
  echo "Querying data"
  cd $HOME_PATH/${DATASET}_testspace/O_10_D_0_0
  mkdir vary_dataRange

  # note
  # enlarge memory allocation for read when querying
  $HOME_PATH/tool.sh write_read_schema_free_memory_proportion 4:3:1:2 $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
  #$HOME_PATH/tool.sh MAX_HEAP_SIZE \"8G\" $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-env.sh

  # attention: case sensitive
  approachArray=("ILTS" "M4" "LTTB" "MinMaxLTTB" "MinMax");
  for approach in ${approachArray[@]};
  do
    echo "[[[[[[[[[[[[[$approach]]]]]]]]]]]]]"

    cd $HOME_PATH/${DATASET}_testspace/O_10_D_0_0/vary_dataRange
    mkdir $approach
    cd $approach
    cp $HOME_PATH/ProcessResult.* .

    # attention: case sensitive enable_Tri
    $HOME_PATH/tool.sh enable_Tri ${approach} $HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties

    # Note the following command print info is appended into result_${per}.txt for query latency exp
    $HOME_PATH/tool.sh REP_ONCE false $HOME_PATH/query_experiment.sh
    find $HOME_PATH -type f -iname "*.sh" -exec chmod +x {} \;

    if [ $approach == "OM3" ]; then
      # device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path om3_query_dir
      $HOME_PATH/query_experiment.sh root "${OM3_MIN_DEVICE}.${MEASUREMENT},${OM3_MAX_DEVICE}.${MEASUREMENT}" \
        ${TIMESTAMP_PRECISION} ${DATA_MIN_TIME} \
        ${DATA_MAX_TIME} ${FIX_QUERY_RANGE} ${FIX_M} $approach \
        "${TRI_VISUALIZATION_EXP}/tools/varyN-byData/ids-${DATASET,,}-${per}.csv" \
        >> result_${per}.txt
    else
      # device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
      $HOME_PATH/query_experiment.sh ${DEVICE} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_MIN_TIME} \
        ${DATA_MAX_TIME} ${FIX_QUERY_RANGE} ${FIX_M} $approach >> result_${per}.txt
    fi

    java ProcessResult result_${per}.txt result_${per}.out ../sumResult_${approach}.csv

  done; # vary approach

done # vary per

######################################################################################
cd $HOME_PATH/${DATASET}_testspace/O_10_D_0_0/vary_dataRange
(cut -f 2,11,12,28,35 -d "," sumResult_MinMax.csv) > tmp1.csv
(cut -f 2,11,12,28,35 -d "," sumResult_M4.csv| paste -d, tmp1.csv -) > tmp2.csv
(cut -f 2,11,12,28,35 -d "," sumResult_LTTB.csv| paste -d, tmp2.csv -) > tmp3.csv
(cut -f 2,11,12,28,35 -d "," sumResult_MinMaxLTTB.csv| paste -d, tmp3.csv -) > tmp4.csv
(cut -f 2,11,12,28,35 -d "," sumResult_ILTS.csv| paste -d, tmp4.csv -) > tmp5.csv

echo "MinMax(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
M4(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
LTTB(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
MinMaxLTTB(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
ILTS(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum"\
 > $HOME_PATH/res-${DATASET}-scalability-byData.csv

sed '1d' tmp5.csv >> $HOME_PATH/res-${DATASET}-scalability-byData.csv
rm tmp*.csv

# add varied parameter value and the corresponding estimated chunks per interval for each line
# estimated chunks per interval = range/m/(totalRange/(pointNum/chunkSize))
# range=totalRange, estimated chunks per interval=(pointNum/chunkSize)/m
sed -i -e 1's/^/range,estimated chunks per interval,/' $HOME_PATH/res-${DATASET}-scalability-byData.csv
line=2
for per in $perlist
do
  range=$((echo scale=0 ; echo ${per}*${TOTAL_TIME_RANGE}/100) | bc )
  c=$((echo scale=0 ; echo ${TOTAL_POINT_NUMBER}/${IOTDB_CHUNK_POINT_SIZE}/${FIX_M}*${per}/100) | bc )
  sed -i -e ${line}"s/^/${range},${c},/" $HOME_PATH/res-${DATASET}-scalability-byData.csv
  let line+=1
done


echo "ALL FINISHED!"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m