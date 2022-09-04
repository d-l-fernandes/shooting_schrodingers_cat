number_gpus=$1
visible_devices=$2
solver=$3

# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/2')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/3')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/4')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/5')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/6')
# bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/7')
bash ./checker.sh $number_gpus $visible_devices $solver $(bc -l <<< '1/8')
