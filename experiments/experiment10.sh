cd ..
# hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --gpus '"device=0,2"' dlf28/pytorch_lightning\
hare run --rm -v "$(pwd)":/app --workdir /app --user $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="4,5" dlf28/pytorch_lightning\
	python main.py \
	  --drift=score_network \
	  --diffusion=scalar \
    --dataset=double_well_right \
    --prior=double_well_left \
    --prior_sde=hill \
	  --prior_dist=gaussian \
	  --batch_size=1500 \
	  --num_epochs=30 \
	  --eval_frequency=100 \
	  --learning_rate=1e-3\
	  --num_steps=20 \
	  --delta_t=0.05 \
	  --num_iter=600 \
	  --sigma=0.001 \
	  --max_gamma=0.3 \
	  --gpus=2
