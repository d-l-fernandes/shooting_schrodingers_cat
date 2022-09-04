number_gpus=$1
visible_devices=$2
solver=$3

bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1')
bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/2')
bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/3') # <=
bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/4')
bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/5')
bash ./hill.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/6')
