approach="M4" # ILTS
m=10000 # 2000
remote_ip=xxx
remote_user_name=xxx
remote_passwd='xxx' # do not use double quotes
HOME_PATH=/root/energyExp

concurrent=10
capacity="8k"
period=200

DEVICE="root.Pyra1.otherDevice"
MEASUREMENT="otherSensor"
TIMESTAMP_PRECISION=ms
DATA_MIN_TIME=0
DATA_MAX_TIME=10000000
FIX_QUERY_RANGE=10000000


cnt=0
while true
do
	SECONDS=0

	date

	let cnt+=1

	tmpDir="tmpDir${cnt}-${approach}-${m}"
	mkdir ${tmpDir}

	for((i=1;i<=${concurrent};i++)) do
		# each corresponds to a device, start from 1
		bash inner.sh ${approach} ${m} ${remote_ip} ${remote_user_name} ${remote_passwd} \
			${DEVICE}${i} ${MEASUREMENT} ${TIMESTAMP_PRECISION} ${DATA_MIN_TIME} ${DATA_MAX_TIME} \
			${FIX_QUERY_RANGE} \
			${cnt} ${tmpDir} ${capacity} \
			${HOME_PATH} \
			2>&1 > log_file-${approach}-${m}-${DEVICE}${i}-${cnt}-query.log &
	done

	wait
	echo "All inner.sh scripts have finished query execution. Elapsed time: ${SECONDS}s"

	remaining_time=$((period - SECONDS))
	echo "sleep ${remaining_time}"
	if [ $remaining_time -gt 0 ]; then
		sleep $remaining_time
	fi

done
