cd ..

number_gpus=$1
visible_devices=$2

# Used in 2021-11-18; 19:05
hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python main.py \
	  --drift=score_network \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=50 \
	  --eval_frequency=20 \
	  --learning_rate=1e-3 \
	  --num_steps=20 \
	  --delta_t=0.05 \
	  --num_iter=10 \
	  --sigma=0.001 \
	  --batch_repeats=20 \
	  --max_gamma=0.3 \
    --gpus="$number_gpus"
