number_gpus=$1
visible_devices=$2

bash ./mine.sh $number_gpus $visible_devices 1.0 0.1
bash ./mine.sh $number_gpus $visible_devices 1.0 0.2
bash ./mine.sh $number_gpus $visible_devices 1.0 0.5
bash ./mine.sh $number_gpus $visible_devices 1.0 1.0

bash ./mine.sh $number_gpus $visible_devices 0.5 0.1
bash ./mine.sh $number_gpus $visible_devices 0.5 0.25
bash ./mine.sh $number_gpus $visible_devices 0.5 0.5

bash ./mine.sh $number_gpus $visible_devices 0.25 0.05
bash ./mine.sh $number_gpus $visible_devices 0.25 0.125
bash ./mine.sh $number_gpus $visible_devices 0.25 0.25

# bash ./mine.sh $number_gpus $visible_devices 0.1 0.05
# bash ./mine.sh $number_gpus $visible_devices 0.1 0.1
