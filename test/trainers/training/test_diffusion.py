import warnings
import os
from src.trainers.trainer import Trainer
from test.trainers.test_diffusion import diffuser

def test_complete_process(diffuser: Trainer):
    if not os.path.exists(diffuser.model_wrapper.model.loss.model_path):
        warnings.warn("KID loss not found. Model can't run.")
    else:
        diffuser.run()
        assert True
