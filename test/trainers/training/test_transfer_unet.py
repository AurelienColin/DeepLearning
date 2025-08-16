from src.trainers.trainer import Trainer
from test.trainers.test_transfer_unet import transferer

def test_complete_process(transferer: Trainer):
    transferer.run()
    assert True
