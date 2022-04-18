number_gpus=$1
visible_devices=$2

# KSD=MEDIAN, DELTA_T=UNIFORM
bash mine.sh "$number_gpus" "$visible_devices" 0.5 "median" True
bash mine.sh "$number_gpus" "$visible_devices" 1.0 "median" True
bash mine.sh "$number_gpus" "$visible_devices" 1.5 "median" True

# KSD=MEDIAN, DELTA_T=RANDOM
bash mine.sh "$number_gpus" "$visible_devices" 0.5 "median" False
bash mine.sh "$number_gpus" "$visible_devices" 1.0 "median" False
bash mine.sh "$number_gpus" "$visible_devices" 1.5 "median" False

# KSD=MEDIAN, DELTA_T=UNIFORM
bash mine.sh "$number_gpus" "$visible_devices" 0.5 "mean" True
bash mine.sh "$number_gpus" "$visible_devices" 1.0 "mean" True
bash mine.sh "$number_gpus" "$visible_devices" 1.5 "mean" True

# KSD=MEDIAN, DELTA_T=RANDOM
bash mine.sh "$number_gpus" "$visible_devices" 0.5 "mean" False
bash mine.sh "$number_gpus" "$visible_devices" 1.0 "mean" False
bash mine.sh "$number_gpus" "$visible_devices" 1.5 "mean" False