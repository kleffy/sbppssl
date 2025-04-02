import os
import sys
# Insert parent directory so soc_enmap can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
model factory for soc estimation pipeline
"""

import torch.nn as nn
from soc_enmap import SOCEnMAPModel, SOCEnMAPSSLModel

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

# Note: Removed the __main__ block.
# Please use a dedicated driver script (outside of __init__.py) to test model creation.