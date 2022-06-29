number_gpus=$1
visible_devices=$2
solver=$3

bash ./checker.sh $number_gpus $visible_devices $solver
bash ./circle.sh $number_gpus $visible_devices $solver
bash ./moon.sh $number_gpus $visible_devices $solver
bash ./s_curve.sh $number_gpus $visible_devices $solver
bash ./swiss_roll.sh $number_gpus $visible_devices $solver
