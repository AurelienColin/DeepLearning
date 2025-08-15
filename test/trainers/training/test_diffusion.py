import warnings

from trainers.trainer import Trainer


def test_complete_process(diffuser: Trainer):
    if not os.path.exists(diffuser.model_wrapper.model.loss.model_path):
        warnings.warn("KID loss not found. Model can't run.")
    else:
        diffuser.run()
        assert True
