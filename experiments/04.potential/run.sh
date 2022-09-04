number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

# bash ./no_prior.sh $number_gpus $visible_devices $solver $do_dsb
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb

# bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb $(bc  -l <<< '1/4')
