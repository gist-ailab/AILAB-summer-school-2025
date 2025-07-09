import sys, os
BASE_DIR = "/workspace/cgnet"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from utils import registry
from cgnet.utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(cfg, **kwargs)
