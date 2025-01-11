#!/bin/bash

export READ_METHOD=raw # raw/lttb/ilts
export TRI_VISUALIZATION_EXP=/root/lts-exp
export remote_TRI_VISUALIZATION_EXP=/root/lts-exp
export remote_IOTDB_HOME_PATH=/root/motivationExp
export remote_ip=127.0.0.1
export remote_user_name=root
export remote_passwd='root' # do not use double quotes

#######################################
# below are local client configurations
export PYTHON_READ_PLOT_PATH=$TRI_VISUALIZATION_EXP/python-exp/python-read-plot.py
export EXPERIMENT_PATH=$TRI_VISUALIZATION_EXP/python-exp/python_query_plot_experiment.sh
export repetition=1
export PROCESS_QUERY_PLOT_JAVA_PATH=$TRI_VISUALIZATION_EXP/python-exp/ProcessQueryPlotResult.java
export tqs=1
export w=480
export local_FILE_PATH=$TRI_VISUALIZATION_EXP/python-exp/localData.csv

# below are remote data server configurations
export remote_IOTDB_SBIN_HOME=$remote_IOTDB_HOME_PATH/iotdb-server-0.12.4/sbin
export remote_IOTDB_CONF_PATH=$remote_IOTDB_HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
export remote_IOTDB_START=$remote_IOTDB_SBIN_HOME/start-server.sh
export remote_IOTDB_STOP=$remote_IOTDB_SBIN_HOME/stop-server.sh
export remote_IOTDB_EXPORT_CSV_TOOL=$remote_IOTDB_HOME_PATH/iotdb-cli-0.12.4/tools
export remote_iotdb_port=6667
export remote_iotdb_username=root
export remote_iotdb_passwd=root
export remote_RAW_FILE_PATH=$remote_IOTDB_HOME_PATH/RTDMORE_TEST/RTDMORE_TEST.csv
export remote_tool_bash=$remote_TRI_VISUALIZATION_EXP/python-exp/tool.sh
export remote_TRI_FILE_PATH=$remote_TRI_VISUALIZATION_EXP/python-exp/TRI.csv

echo "begin"

# prepare ProcessQueryPlotResult tool
sed '/^package/d' ProcessQueryPlotResult.java > ProcessQueryPlotResult2.java
rm ProcessQueryPlotResult.java
mv ProcessQueryPlotResult2.java ProcessQueryPlotResult.java
javac ProcessQueryPlotResult.java

for N in 50000000 100000000 150000000 200000000 250000000 300000000
do
	echo "N=$N"
	export N=$N
  export tqe=$N # uniform timestamps in this case

	$EXPERIMENT_PATH >result-${READ_METHOD}_${N}.txt #> is overwrite, >> is append

	java ProcessQueryPlotResult result-${READ_METHOD}_${N}.txt result-${READ_METHOD}_${N}.out sumResult-${READ_METHOD}.csv ${N}
done

echo "ALL FINISHED!"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m