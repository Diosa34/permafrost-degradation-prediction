$ErrorActionPreference = "Stop"

# MLflow UI must point to the same backend store as training.
# In this project we use sqlite by default (see ml/train.py).

$backendStore = "sqlite:///mlflow.db"
$port = 5000

Write-Host "Starting MLflow UI on http://127.0.0.1:$port"
Write-Host "Backend store: $backendStore"

python -m mlflow ui --backend-store-uri $backendStore --host 127.0.0.1 --port $port

