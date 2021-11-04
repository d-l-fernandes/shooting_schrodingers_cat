cd ..
# hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --gpus '"device=0,2"' dlf28/pytorch_lightning\
hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="3" dlf28/pytorch_lightning\
	python main.py \
	  --drift=score_network \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --schedule=constant \
	  --batch_size=1500 \
	  --num_epochs=30 \
	  --eval_frequency=200 \
	  --learning_rate=1e-2 \
	  --num_steps=10 \
	  --delta_t=0.05 \
	  --num_iter=100 \
	  --sigma=0.001 \
	  --max_gamma=0.5 \
	  --gpus=1
