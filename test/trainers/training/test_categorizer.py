from trainers.trainer import Trainer


def test_complete_process(categorizer: Trainer):
    categorizer.run()
    assert True
