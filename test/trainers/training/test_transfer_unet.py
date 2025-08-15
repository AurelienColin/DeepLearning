from trainers.trainer import Trainer


def test_complete_process(transferer: Trainer):
    transferer.run()
    assert True
