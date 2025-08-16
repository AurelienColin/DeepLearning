from src.trainers.trainer import Trainer
from test.trainers.test_categorizer import categorizer

def test_complete_process(categorizer: Trainer):
    categorizer.run()
    assert True
