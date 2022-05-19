number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

bash ./checker.sh $number_gpus $visible_devices $solver $do_dsb
bash ./circle.sh $number_gpus $visible_devices $solver $do_dsb
bash ./moon.sh $number_gpus $visible_devices $solver $do_dsb
bash ./s_curve.sh $number_gpus $visible_devices $solver $do_dsb
bash ./swiss_roll.sh $number_gpus $visible_devices $solver $do_dsb
