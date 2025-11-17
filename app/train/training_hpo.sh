python -m app.train.training_hpo \
    --n_trials 2 \
    --study_name "dogs-efficientnet-hpo-v1" \
    --optuna_db "optuna_dogs.db"
