number_gpus=$1
visible_devices=$2
solver=$3
do_dsb=${4:-False}

# bash ./no_prior.sh $number_gpus $visible_devices $solver $do_dsb
# bash ./bimodal.sh $number_gpus $visible_devices $solver $do_dsb

bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 1.
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 0.5
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 0.33333
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 0.25
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 0.2
bash ./hill.sh $number_gpus $visible_devices $solver $do_dsb 0.1
