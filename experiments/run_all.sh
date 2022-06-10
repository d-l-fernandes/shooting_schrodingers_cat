number_gpus=$1
visible_devices=$2
solver=$3

cd ./01.no_prior/
bash run.sh $number_gpus $visible_devices $solver
cd ..
cd ./02.brownian_prior/
bash run.sh $number_gpus $visible_devices $solver
cd ..
cd ./03.blobs/
bash run.sh $number_gpus $visible_devices $solver
cd ..
cd ./04.well/
bash hill.sh $number_gpus $visible_devices $solver
