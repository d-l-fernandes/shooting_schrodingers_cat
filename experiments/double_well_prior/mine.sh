cd ../../

number_gpus=$1
visible_devices=$2
gamma=$3
ksd_scale=$4
uniform_delta_t=$5

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python -W ignore main.py \
	  --drift=nn_general \
      --dataset=double_well_right \
      --prior=double_well_left \
      --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=15 \
	  --learning_rate=2e-4 \
	  --num_steps=25 \
	  --sigma=1e-3 \
	  --total_gamma="$gamma" \
	  --normalize=False \
	  --solver=srk \
      --gpus="$number_gpus" \
      --uniform_delta_t="$uniform_delta_t" \
	  --ksd_scale="$ksd_scale"
