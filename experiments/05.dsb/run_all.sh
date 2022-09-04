number_gpus=$1
visible_devices=$2

bash ./run_short.sh $number_gpus $visible_devices
bash ./run_long.sh $number_gpus $visible_devices
bash ./run_extra_long.sh $number_gpus $visible_devices
