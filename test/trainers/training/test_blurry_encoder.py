from src.trainers.trainer import Trainer
from test.trainers.test_blurry_encoder import blurry_encoder

def test_complete_process(blurry_encoder: Trainer):
    blurry_encoder.run()
    assert True
