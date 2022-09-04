number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

bash ./brownian.sh $number_gpus $visible_devices $solver $do_dsb
bash ./whirlpool.sh $number_gpus $visible_devices $solver $do_dsb
bash ./eye_of_sauron.sh $number_gpus $visible_devices $solver $do_dsb
