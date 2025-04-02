import torch.nn as nn
from .soc_enmap import SOCEnMAPModel, SOCEnMAPSSLModel

MODEL_REGISTRY = {
    'socenmap': {
        'main': SOCEnMAPModel,
        'ssl': SOCEnMAPSSLModel,
    },
}

def create_model(model_name, is_ssl=False, **kwargs):
    """
    Create a model based on the model name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_dict = MODEL_REGISTRY[model_name]
    model_type = 'ssl' if is_ssl else 'main'
    
    if model_type not in model_dict:
        raise ValueError(f"Model type {model_type} not found in registry for model {model_name}. Available types: {list(model_dict.keys())}")
    
    model_class = model_dict[model_type]
    return model_class(**kwargs)

# Note: The __main__ block was removed. Use the dedicated driver script for testing.