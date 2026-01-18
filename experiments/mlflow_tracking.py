import mlflow

def start_experiment(name):
    mlflow.set_experiment(name)
    mlflow.start_run()

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
