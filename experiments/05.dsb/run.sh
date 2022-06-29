number_gpus=$1
visible_devices=$2

# Normal experiments
bash ./checker.sh $number_gpus $visible_devices
bash ./circle.sh $number_gpus $visible_devices
bash ./moon.sh $number_gpus $visible_devices
bash ./s_curve.sh $number_gpus $visible_devices
bash ./swiss_roll.sh $number_gpus $visible_devices

# Blobs experiments
bash ./brownian.sh $number_gpus $visible_devices
bash ./whirlpool.sh $number_gpus $visible_devices
bash ./menorah.sh $number_gpus $visible_devices

# Hill experiment
bash ./hill.sh $number_gpus $visible_devices
