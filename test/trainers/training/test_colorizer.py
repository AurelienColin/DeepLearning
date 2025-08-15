from trainers.trainer import Trainer


def test_complete_process(colorizer: Trainer):
    colorizer.run()
    assert True
