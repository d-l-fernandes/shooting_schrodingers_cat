cd ..

number_gpus=$1
visible_devices=$2

hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$visible_devices" dlf28/pytorch_lightning \
	python main.py \
	  --drift=score_network \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=25 \
	  --eval_frequency=600 \
	  --learning_rate=1e-3 \
	  --num_steps=100 \
	  --delta_t=0.01 \
	  --num_iter=300 \
	  --sigma=0.001 \
	  --max_gamma=0.3 \
    --gpus="$number_gpus"\
    --do_dsb
