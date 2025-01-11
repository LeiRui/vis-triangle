BASE_PATH=/root

TRI_VISUALIZATION_EXP=${BASE_PATH}/lts-exp
HOME_PATH=${BASE_PATH}/cacheExp

VALUE_ENCODING=PLAIN
TIME_ENCODING=PLAIN
COMPRESSOR=UNCOMPRESSED
DATA_TYPE=double

mkdir -p $HOME_PATH

find $TRI_VISUALIZATION_EXP -type f -iname "*.sh" -exec chmod +x {} \;
find $TRI_VISUALIZATION_EXP -type f -iname "*.sh" -exec sed -i -e 's/\r$//' {} \;

# check bc installed
REQUIRED_PKG="bc"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

#====prepare general environment====
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/tools/tool.sh .
cp $TRI_VISUALIZATION_EXP/jars/WriteDataMore*.jar .
cp $TRI_VISUALIZATION_EXP/jars/QueryData*.jar .
# Due to the large size of this experiments.jar,
# it has not been directly placed in the Git repository, but is instead in datasets
cp $TRI_VISUALIZATION_EXP/datasets/experiments.jar .
cp $TRI_VISUALIZATION_EXP/tools/query_experiment.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH $HOME_PATH/query_experiment.sh
scp -r $TRI_VISUALIZATION_EXP/iotdb-server-0.12.4 .
scp -r $TRI_VISUALIZATION_EXP/iotdb-cli-0.12.4 .
cp $TRI_VISUALIZATION_EXP/tools/iotdb-engine-example.properties .
cp $TRI_VISUALIZATION_EXP/tools/ProcessResult.java .
cp $TRI_VISUALIZATION_EXP/tools/SumResultUnify.java .
# remove the line starting with "package" in the java file
sed '/^package/d' ProcessResult.java > ProcessResult2.java
rm ProcessResult.java
mv ProcessResult2.java ProcessResult.java
# then javac it
javac ProcessResult.java
# remove the line starting with "package" in the java file
sed '/^package/d' SumResultUnify.java > SumResultUnify2.java
rm SumResultUnify.java
mv SumResultUnify2.java SumResultUnify.java
# then javac it
javac SumResultUnify.java


#############################################
#========prepare for motivation exp========
#############################################
NAMES=("cache");
echo "prepare bashes";
for value in ${NAMES[@]};
do
  cd $HOME_PATH
  cp $TRI_VISUALIZATION_EXP/bash/run-${value}-exp.sh .
  $HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-${value}-exp.sh
  $HOME_PATH/tool.sh DATASET Qloss run-${value}-exp.sh
  $HOME_PATH/tool.sh DEVICE "root.Qloss.targetDevice" run-${value}-exp.sh
  $HOME_PATH/tool.sh MEASUREMENT "test" run-${value}-exp.sh
  $HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-${value}-exp.sh
  $HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-${value}-exp.sh
  $HOME_PATH/tool.sh DATA_MIN_TIME 0 run-${value}-exp.sh
  $HOME_PATH/tool.sh DATA_MAX_TIME 10000000 run-${value}-exp.sh
  $HOME_PATH/tool.sh TOTAL_POINT_NUMBER 10000000 run-${value}-exp.sh
  $HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-${value}-exp.sh
  $HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-${value}-exp.sh
  $HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-${value}-exp.sh
  $HOME_PATH/tool.sh accuracy 0.95 run-${value}-exp.sh
  $HOME_PATH/tool.sh querySelectivity 0.1 run-${value}-exp.sh
  $HOME_PATH/tool.sh prefetching 1 run-${value}-exp.sh
  $HOME_PATH/tool.sh aggFactorMinMax 4 run-${value}-exp.sh
  $HOME_PATH/tool.sh realT false run-${value}-exp.sh
  $HOME_PATH/tool.sh seqCount 50 run-${value}-exp.sh
  $HOME_PATH/tool.sh width 1000 run-${value}-exp.sh
  $HOME_PATH/tool.sh height 250 run-${value}-exp.sh
  $HOME_PATH/tool.sh TRI_VISUALIZATION_EXP ${TRI_VISUALIZATION_EXP} run-${value}-exp.sh
  $HOME_PATH/tool.sh INPUT_DATA_PATH ${TRI_VISUALIZATION_EXP}/datasets/Qloss.csv run-${value}-exp.sh
done;

echo "prepare testspace directory";
cd $HOME_PATH
mkdir Qloss_testspace


find $HOME_PATH -type f -iname "*.sh" -exec chmod +x {} \;

echo "finish"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m
