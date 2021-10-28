cd ..
python main.py \
  --drift=score_network \
  --diffusion=scalar \
  --dataset=s_curve \
  --prior=gaussian \
  --prior_sde=brownian \
  --prior_dist=gaussian \
  --batch_size=50 \
  --num_epochs=20 \
  --eval_frequency=80 \
  --learning_rate=1e-3\
  --num_steps=10 \
  --delta_t=0.05 \
  --num_iter=40 \
  --sigma=0.001 \
