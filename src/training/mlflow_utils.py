"""
MLflow Utilities
Handles experiment tracking, model logging, and model registry operations.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional


def setup_mlflow(
    tracking_uri: str = None,
    experiment_name: str = "movie-recommender",
) -> str:
    """
    Initialize MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    except Exception as e:
        print(f"Tracking server at '{tracking_uri}' unavailable ({e}). Falling back to 'sqlite:///mlflow.db'")
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_run(
    model,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    model_path: str,
    run_name: str = None,
    tags: Dict[str, str] = None,
) -> str:
    """
    Log a complete training run to MLflow.
    
    Args:
        model: Trained model object
        metrics: Evaluation metrics dictionary
        params: Model hyperparameters
        model_path: Path where model artifacts are saved
        run_name: Optional name for the run
        tags: Optional tags for the run
        
    Returns:
        MLflow run ID
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Log model artifacts
        if os.path.exists(model_path):
            mlflow.log_artifacts(model_path, artifact_path="model")

        run_id = run.info.run_id
        print(f"MLflow run logged: {run_id}")

        return run_id


def register_model(
    run_id: str,
    model_name: str = "hybrid-recommender",
    artifact_path: str = "model",
) -> int:
    """
    Register a model version in MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path within the run's artifacts
        
    Returns:
        Model version number
    """
    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        result = mlflow.register_model(model_uri, model_name)
        version = int(result.version)
        print(f"Registered model '{model_name}' version {version}")
        return version
    except Exception as e:
        # Fallback to direct client registration
        try:
            try:
                client.create_registered_model(model_name)
            except Exception:
                pass
            run = client.get_run(run_id)
            source = f"{run.info.artifact_uri}/{artifact_path}"
            mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
            version = int(mv.version)
            print(f"Registered model '{model_name}' version {version} via MLflowClient")
            return version
        except Exception as inner_e:
            print(f"Error registering model: {inner_e}")
            raise inner_e


def promote_model(
    model_name: str,
    version: int,
    stage: str = "Production",
) -> bool:
    """
    Promote a model version to a stage (Staging/Production).
    
    Args:
        model_name: Registered model name
        version: Version number to promote
        stage: Target stage
        
    Returns:
        True if promotion succeeded
    """
    client = MlflowClient()

    try:
        # Transition to requested stage
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=True,
        )
        print(f"Model '{model_name}' v{version} promoted to {stage}")
        return True
    except Exception as e:
        print(f"Error promoting model: {e}")
        return False


def get_production_model_version(
    model_name: str = "hybrid-recommender",
) -> Optional[int]:
    """Get the current production model version."""
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return int(versions[0].version)
        return None
    except Exception:
        return None


def get_model_metrics(run_id: str) -> Dict[str, float]:
    """Get metrics from an MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}


def compare_models(
    candidate_metrics: Dict[str, float],
    production_metrics: Dict[str, float],
    primary_metric: str = "ndcg_at_10",
    min_improvement: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare candidate model against production model.
    
    Args:
        candidate_metrics: Metrics from candidate model
        production_metrics: Metrics from current production model
        primary_metric: Primary metric for comparison
        min_improvement: Minimum required improvement ratio
        
    Returns:
        Comparison result with promote recommendation
    """
    candidate_value = candidate_metrics.get(primary_metric, 0)
    production_value = production_metrics.get(primary_metric, 0)

    if production_value > 0:
        improvement = (candidate_value - production_value) / production_value
    else:
        improvement = 1.0 if candidate_value > 0 else 0.0

    should_promote = improvement >= min_improvement

    result = {
        "primary_metric": primary_metric,
        "candidate_value": candidate_value,
        "production_value": production_value,
        "improvement": improvement,
        "should_promote": should_promote,
        "all_candidate_metrics": candidate_metrics,
        "all_production_metrics": production_metrics,
    }

    status = "✅ PROMOTE" if should_promote else "❌ REJECT"
    print(f"\nModel comparison ({primary_metric}): {status}")
    print(f"  Candidate: {candidate_value:.4f}")
    print(f"  Production: {production_value:.4f}")
    print(f"  Improvement: {improvement:+.2%}")

    return result
