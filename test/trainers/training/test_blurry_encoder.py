from trainers.trainer import Trainer


def test_complete_process(blurry_encoder: Trainer):
    blurry_encoder.run()
    assert True
