## this file will fetch the best model, register it and save it locally

import mlflow

experiment_id = mlflow.get_experiment_by_name()

best_run_df = mlflow.search_runs(
    experiment_ids = [experiment_id]
)