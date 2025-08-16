from src.trainers.trainer import Trainer
from test.trainers.test_colorizer import colorizer

def test_complete_process(colorizer: Trainer):
    colorizer.run()
    assert True
