# MLflow Notes

- Set `MLFLOW_TRACKING_URI` to a server or local folder.
- Log params/metrics in training script where marked (add mlflow.start_run()).
- Register best model with `mlflow.pytorch.log_model()` for lineage.
