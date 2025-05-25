# import os.path
# import shutil
#
# from src.trainers.trainer import Trainer
#
#
# def run(trainer: Trainer) -> None:
#     trainer.epochs = 1
#     trainer.training_steps = 5
#     trainer.validation_steps = 5
#     trainer.model_wrapper._output_folder = 'test/.tmp/trainer'
#     if os.path.exists(trainer.model_wrapper.output_path):
#         shutil.rmtree(trainer.model_wrapper.output_path)
#
#     trainer.run()
#
#
# def test_colorizer_gochiusa():
#     from src.trainers.image_to_image_trainers.run.gochiusa_colorizer import trainer
#     run(trainer)
#
#
# def test_gochiusa_blurry_encoder():
#     from src.trainers.image_to_image_trainers.run.gochiusa_blurry_encoder import trainer
#     run(trainer)
#
#
# def test_gochiusa_diffusion():
#     from src.trainers.image_to_image_trainers.run.gochiusa_diffusion import trainer
#     run(trainer)
#
#
# def test_uncensor_unet():
#     from src.trainers.image_to_image_trainers.run.uncensor_unet import trainer
#     run(trainer)
#
#
# def test_gochisa_categorizer():
#     from src.trainers.image_to_tag_trainers.run.gochiusa_categorizer import trainer
#     run(trainer)
#
#
# def test_rating_tagger():
#     from src.trainers.image_to_tag_trainers.run.rating_tagger import trainer
#     run(trainer)
#
#
# def test_clothe_saliency():
#     from src.trainers.image_to_image_trainers.run.clothe_saliency import trainer
#     run(trainer)
