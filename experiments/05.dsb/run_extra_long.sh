number_gpus=$1
visible_devices=$2


# Normal experiments
bash ./checker.sh $number_gpus $visible_devices 200
bash ./circle.sh $number_gpus $visible_devices 200
bash ./moon.sh $number_gpus $visible_devices 200
bash ./s_curve.sh $number_gpus $visible_devices 200
bash ./swiss_roll.sh $number_gpus $visible_devices 200

# Blobs experiments
bash ./brownian.sh $number_gpus $visible_devices 200
bash ./whirlpool.sh $number_gpus $visible_devices 200
bash ./eye_of_sauron.sh $number_gpus $visible_devices 200

# Hill experiment
bash ./hill.sh $number_gpus $visible_devices 200
