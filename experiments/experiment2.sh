cd ..
python main.py \
  --drift=nn_general \
  --diffusion=scalar \
  --dataset=toy_experiment_blobs_3d \
  --dims=3 \
  --prior=gaussian \
  --prior_sde=brownian \
  --prior_dist=gaussian \
  --batch_size=50 \
  --num_epochs=30 \
  --eval_frequency=100 \
  --learning_rate=1e-3\
  --num_steps=10 \
  --delta_t=0.05 \
  --num_iter=50 \
  --batch_repeats=1 \
  --sigma=0.0001 \
