number_gpus=$1
visible_devices=$2

cd ./04.potential/
bash hill.sh $number_gpus $visible_devices srk
bash hill.sh $number_gpus $visible_devices em
cd ..
cd ./03.blobs/
bash run.sh $number_gpus $visible_devices srk
bash run.sh $number_gpus $visible_devices em
cd ..
cd ./02.brownian_prior/
bash run.sh $number_gpus $visible_devices srk
bash run.sh $number_gpus $visible_devices em
# cd ..
# cd ./01.no_prior/
# bash run.sh $number_gpus $visible_devices srk
# bash run.sh $number_gpus $visible_devices em
