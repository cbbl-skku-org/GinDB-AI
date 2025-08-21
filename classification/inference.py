import torch
import numpy as np
import os
import copy
from typing import Tuple, List
import logging
from architectures import SimpleNeuralNetwork
from PyFingerprint.fingerprint import get_fingerprint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    """
    Inference class for deploying trained models on server.
    Loads all fold models and performs ensemble prediction by averaging probabilities.
    """
    
    def __init__(self, checkpoint_dir: str, args_config: dict, device: str = 'cuda'):
        """
        Initialize the inference class.
        
        Args:
            checkpoint_dir (str): Directory containing all fold checkpoint files (.ckpt)
            args_config (dict): Configuration dictionary containing model parameters
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.args_config = args_config
        self.models = []
        
        # Load all models
        self._load_models()
        
        logger.info(f"Initialized inference with {len(self.models)} models on {self.device}")
    
    def _load_models(self) -> None:
        """
        Load all fold models from checkpoint directory.
        """
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.ckpt')]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.checkpoint_dir}")
        
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_file)
            
            # Create model instance
            model = self._create_model()
            
            # Load state dict
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)['state_dict']
                
                # Remove prefix 'model.'
                state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                model.to(self.device)
                model = model.float()
                
                # Store a deep copy to avoid reference issues
                self.models.append(copy.deepcopy(model))
                
                logger.info(f"Loaded model from {ckpt_file}")
                
            except Exception as e:
                logger.error(f"Failed to load model from {ckpt_file}: {str(e)}")
                raise
    
    def _create_model(self):
        """
        Create a model instance based on the configuration.
        """
        # Create a simple args object from config dict
        class Args:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        args = Args(self.args_config)
        
        # Create the neural network (assuming SimpleNeuralNetwork from your architectures module)
        model = SimpleNeuralNetwork(
            args=args,
            num_classes=args.num_classes,
            embeddings_dim=args.embed_dim
        )
        
        return model
    
    def predict(self, input_array: np.ndarray) -> Tuple[float, int]:
        """
        Perform inference on input array.
        
        Args:
            input_array (np.ndarray): Input embedding array of shape (embed_dim,)
            
        Returns:
            Tuple[float, int]: (probability_of_positive_class, predicted_label)
        """
        if len(input_array.shape) == 1:
            # Add batch dimension if single sample
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().to(self.device)
        else:
            input_tensor = torch.from_numpy(input_array).float().to(self.device)
        
        return self._predict_tensor(input_tensor)
    
    def predict_batch(self, input_arrays: List[np.ndarray]) -> List[Tuple[float, int]]:
        """
        Perform batch inference on multiple input arrays.
        
        Args:
            input_arrays (List[np.ndarray]): List of input embedding arrays
            
        Returns:
            List[Tuple[float, int]]: List of (probability_of_positive_class, predicted_label) tuples
        """
        # Stack arrays into batch tensor
        batch_tensor = torch.stack([torch.from_numpy(arr) for arr in input_arrays])
        batch_tensor = batch_tensor.float().to(self.device)
        
        results = []
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        for i in range(0, batch_tensor.shape[0], batch_size):
            batch_slice = batch_tensor[i:i+batch_size]
            batch_results = self._predict_tensor(batch_slice)
            
            if isinstance(batch_results, tuple):
                # Single result
                results.append(batch_results)
            else:
                # Multiple results
                results.extend(batch_results)
        
        return results
    
    def predict_sequence(self, sequence: str) -> Tuple[float, int]:
        """
        Perform inference on a molecular sequence string.
        
        Args:
            sequence (str): Molecular sequence string
            
        Returns:
            Tuple[float, int]: (probability_of_positive_class, predicted_label)
        """
        try:
            # Extract features from sequence using PyFingerprint
            feature = get_fingerprint(sequence, 'mol2vec').to_numpy()
            
            # Perform prediction
            return self.predict(feature)
            
        except Exception as e:
            logger.error(f"Failed to process sequence: {str(e)}")
            raise RuntimeError(f"Sequence processing failed: {str(e)}")
    
    def predict_sequences_batch(self, sequences: List[str]) -> List[Tuple[float, int]]:
        """
        Perform batch inference on multiple molecular sequence strings.
        
        Args:
            sequences (List[str]): List of molecular sequence strings
            
        Returns:
            List[Tuple[float, int]]: List of (probability_of_positive_class, predicted_label) tuples
        """
        features = []
        failed_indices = []
        
        for i, sequence in enumerate(sequences):
            try:
                feature = get_fingerprint(sequence, 'mol2vec').to_numpy()
                features.append(feature)
            except Exception as e:
                logger.warning(f"Failed to process sequence {i}: {str(e)}")
                failed_indices.append(i)
                features.append(None)
        
        # Filter out failed sequences
        valid_features = [f for f in features if f is not None]
        
        if not valid_features:
            raise RuntimeError("All sequences failed to process")
        
        # Perform batch prediction on valid features
        valid_results = self.predict_batch(valid_features)
        
        # Reconstruct results with None for failed sequences
        results = []
        valid_idx = 0
        
        for i in range(len(sequences)):
            if i in failed_indices:
                results.append((None, None))
            else:
                results.append(valid_results[valid_idx])
                valid_idx += 1
        
        return results
    
    def _predict_tensor(self, input_tensor: torch.Tensor) -> Tuple[float, int]:
        """
        Internal method to perform prediction on tensor input.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, embed_dim)
            
        Returns:
            Tuple[float, int] or List[Tuple[float, int]]: Prediction results
        """
        if not self.models:
            raise RuntimeError("No models loaded for inference")
        
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(input_tensor)
                prob = output.softmax(dim=-1)
                all_probs.append(prob)
        
        # Average probabilities across all models
        avg_prob = torch.stack(all_probs).mean(dim=0)
        
        # Get predictions
        predicted_labels = torch.argmax(avg_prob, dim=-1)
        positive_class_probs = avg_prob[:, 1]  # Probability of class 1
        
        # Convert to Python types
        if input_tensor.shape[0] == 1:
            # Single prediction
            return (
                float(positive_class_probs[0].cpu().numpy()),
                int(predicted_labels[0].cpu().numpy())
            )
        else:
            # Batch prediction
            return [
                (float(prob.cpu().numpy()), int(label.cpu().numpy()))
                for prob, label in zip(positive_class_probs, predicted_labels)
            ]
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded models.
        
        Returns:
            dict: Model information
        """
        return {
            'num_models': len(self.models),
            'device': self.device,
            'checkpoint_dir': self.checkpoint_dir,
            'config': self.args_config
        }


# Example usage and configuration
def create_inference_instance(checkpoint_dir: str, model_type: str = 'NO_FUNC_YES_FUNC') -> ModelInference:
    """
    Factory function to create inference instance with proper configuration.
    
    Args:
        checkpoint_dir (str): Path to checkpoint directory
        model_type (str): Type of model ('NO_FUNC_YES_FUNC' or 'SINGLE_MULTI')
        
    Returns:
        ModelInference: Configured inference instance
    """
    
    # Example configuration - adjust based on your trained models
    config = {
        'num_classes': 2,
        'embed_dim': 300,
        'number_of_layers': 5,  # Adjust based on your model
        'norm_type': 'batchnorm',   # Adjust based on your model
        'dropout': 0.0,
        'alpha_focal': [0.5, 0.5],   # Adjust based on your model
        'beta_focal': 1,        # Adjust based on your model
        'train_no_func_yes_func': model_type == 'NO_FUNC_YES_FUNC'
    }
    
    return ModelInference(checkpoint_dir, config)


if __name__ == "__main__":
    # Example usage
    
    # Initialize inference
    checkpoint_dir = "/home/cbbl2/Documents/codes/ginseng/GinDB-AI/classification/checkpoints/"
    inference = create_inference_instance(checkpoint_dir)
    
    
    print(inference.predict_sequences_batch(["C=C(C)C(O)CCC(C)(OC1OC(COC2OCC(O)C(O)C2O)C(O)C(O)C1O)C1CCC2(C)C1C(O)CC1C3(C)CC(O)C(OC4OC(CO)C(O)C(O)C4OC4OC(CO)C(O)C(O)C4O)C(C)(C)C3CCC12C", "CCCCCCC[C@H](O[H])/C([H])=C([H])/C#CC#C[C@@H](O[H])C=C"]))