number_gpus=$1
visible_devices=$2

# bash mine.sh "$number_gpus" "$visible_devices" 0.5
bash mine.sh "$number_gpus" "$visible_devices" 1.0
bash mine.sh "$number_gpus" "$visible_devices" 1.5
