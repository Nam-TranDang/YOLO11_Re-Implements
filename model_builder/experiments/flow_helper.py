import mlflow
import wandb
from typing import Dict, Any
import torch
import numpy as np
from pathlib import Path
import os

class ExperimentTracker:
    def __init__(self, config: Dict[str, Any], use_mlflow: bool = True, use_wandb: bool = True):
        self.config = config
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        
        if self.use_mlflow:
            self._init_mlflow()
        if self.use_wandb:
            self._init_wandb()
    
    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        experiment_name = os.getenv('EXPERIMENT_NAME', 'ocr_experiment')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = mlflow.create_experiment(experiment_name)
            
        mlflow.start_run(experiment_id=experiment_id)
        
        # Log config parameters
        mlflow.log_params({
            'detection_model': self.config['detection_model']['name'],
            'recognition_model': self.config['recognition_model']['name'],
            'batch_size': self.config['training']['detection']['batch_size'],
            'learning_rate_detection': self.config['training']['detection']['lr'],
            'learning_rate_recognition': self.config['training']['recognition']['lr'],
            'epochs_detection': self.config['training']['detection']['epochs'],
            'epochs_recognition': self.config['training']['recognition']['epochs']
        })
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        project_name = os.getenv('PROJECT_NAME', 'yolo11_text')
        
        wandb.init(
            project=project_name,
            config=self.config,
            name=f"ocr_run_{wandb.util.generate_id()}"
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to tracking services"""
        if self.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, model_name: str):
        """Log model artifacts"""
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, model_name)
        
        if self.use_wandb:
            # Save model checkpoint
            checkpoint_path = f"checkpoints/{model_name}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
    
    def log_image(self, image: np.ndarray, name: str, step: int = None):
        """Log images"""
        if self.use_wandb:
            wandb.log({name: wandb.Image(image)}, step=step)
    
    def log_artifacts(self, artifact_dir: str):
        """Log artifacts directory"""
        if self.use_mlflow:
            mlflow.log_artifacts(artifact_dir)
        
        if self.use_wandb:
            wandb.save(f"{artifact_dir}/*")
    
    def finish(self):
        """Finish tracking"""
        if self.use_mlflow:
            mlflow.end_run()
        
        if self.use_wandb:
            wandb.finish()

# Usage example in trainer
def create_experiment_tracker(config):
    """Factory function to create experiment tracker"""
    return ExperimentTracker(
        config=config,
        use_mlflow=config.get('experiments', {}).get('use_mlflow', True),
        use_wandb=config.get('experiments', {}).get('use_wandb', True)
    )
