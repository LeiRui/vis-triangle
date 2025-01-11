approach=$1
m=$2

remote_ip=$3
remote_user_name=$4
remote_passwd=$5

DEVICE=$6
MEASUREMENT=$7
TIMESTAMP_PRECISION=$8
DATA_MIN_TIME=$9
DATA_MAX_TIME=${10}
FIX_QUERY_RANGE=${11}

cnt=${12}
tmpDir=${13}

capacity=${14}

HOME_PATH=${15}

start=$(date +%s.%N)

java -jar ${HOME_PATH}/EnergyExpQuery*.jar ${DEVICE} ${MEASUREMENT} \
	${TIMESTAMP_PRECISION} ${DATA_MIN_TIME} ${DATA_MAX_TIME} ${FIX_QUERY_RANGE} \
	$m $approach true ${tmpDir}/${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}.csv

end=$(date +%s.%N)
echo "Query took $(echo "$end - $start" | bc) seconds."

gzip -kf ${tmpDir}/${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}.csv
ls -lh ${tmpDir}/${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}.csv.gz

split -b ${capacity} ${tmpDir}/${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}.csv.gz \
	${tmpDir}/part_${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}_

prefix=${tmpDir}/part_${DEVICE}-${MEASUREMENT}-${approach}-${m}-${cnt}_
suffix="*"
(
	for file in ${prefix}${suffix}; do
		if [ -f "$file" ]; then
			echo "Transferring $file ..."
			sshpass -p "${remote_passwd}" scp ${file} ${remote_user_name}@${remote_ip}:~/.
		fi
	done
) 2>&1 > log_file-${approach}-${m}-${DEVICE}-${cnt}-transfer.log &

echo 3 | sudo tee /proc/sys/vm/drop_caches
sleep 3s

end=$(date +%s.%N)
echo "Total took $(echo "$end - $start" | bc) seconds."
