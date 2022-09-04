number_gpus=$1
visible_devices=$2

cd ./02.brownian_prior/
bash circle.sh $number_gpus $visible_devices em
bash moon.sh $number_gpus $visible_devices em
cd ..

cd ./03.blobs/
bash eye_of_sauron.sh $number_gpus $visible_devices rossler
cd ..

cd ./04.potential/
bash hill.sh $number_gpus $visible_devices em
bash hill.sh $number_gpus $visible_devices rossler
bash hill.sh $number_gpus $visible_devices srk
