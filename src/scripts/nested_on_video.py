import argparse
import os
import typing
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
from rignak.src.lazy_property import LazyProperty
from rignak.src.logging_utils import logger
from src.trainers.image_to_tag_trainers.run.nested_categorizer import DanbooruNestedCategorizerTrainer
import shutil


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--video_filename", type=str, required=True)
    return parser.parse_args()


@dataclass
class Processor:
    model_folder: str
    video_filename: str
    frame_rate: int = 1

    _trainer: typing.Optional[DanbooruNestedCategorizerTrainer] = None

    @property
    def temp_folder(self) -> str:
        return f".tmp/{os.path.basename(self.video_filename)}/frames"

    @LazyProperty
    def trainer(self) -> DanbooruNestedCategorizerTrainer:
        trainer = DanbooruNestedCategorizerTrainer(on_start=False)
        trainer.model_wrapper.model.load_weights(self.model_folder + '/model.h5')
        return trainer

    def extract_frames(self) -> None:
        logger(f"Extracting frames from {self.video_filename}", indent=1)
        os.makedirs(self.temp_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_filename)
        video_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.set_iterator(frame_count)

        frame_nb = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_nb % video_frame_rate == 0:
                resized_frame = cv2.resize(frame, self.trainer.input_shape[:2])
                frame_filename = f"{self.temp_folder}/{frame_nb}.jpg"
                cv2.imwrite(frame_filename, resized_frame)

            frame_nb += 1
            logger.iterate()

        cap.release()
        logger(f"Extracted {len(os.listdir(self.temp_folder))} frames to {self.temp_folder}", indent=-1)

    def predict_tags(self) -> np.ndarray:
        logger("Predicting tags", indent=1)

        frame_filenames = sorted(os.listdir(self.temp_folder), key=lambda x: int(x.split('.')[0]))
        frame_paths = [os.path.join(self.temp_folder, f) for f in frame_filenames]

        predictions = []

        logger.set_iterator(len(frame_paths))

        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            image = image / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            prediction = self.trainer.model_wrapper.model.predict(image, verbose=0)
            predictions.append(np.hstack(prediction))
            logger.iterate()

        logger("Predicting tags OK", indent=-1)
        return np.vstack(predictions)

    def aggregate_results(self, predictions: np.ndarray) -> None:
        """
        Aggregate the predictions into a CSV file.
        """
        logger("Aggregating results", indent=1)

        output_space = self.trainer.output_space
        columns = [f"{category.name}_{label}" for category in output_space.categories for label in category.labels]

        indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in
                   sorted(os.listdir(self.temp_folder), key=lambda x: int(x.split('.')[0]))]

        df = pd.DataFrame(predictions, columns=columns, index=indices)

        output_filename = self.video_filename.replace(os.path.splitext(self.video_filename)[1], ".csv")
        df.to_csv(output_filename, index_label="frame")

        logger(f"Saved results to {output_filename}", indent=-1)

    def cleanup(self) -> None:
        """
        Remove the temporary folder.
        """
        logger(f"Cleaning up temporary folder {self.temp_folder}", indent=1)
        shutil.rmtree(self.temp_folder)
        logger("Cleanup OK", indent=-1)

    def run(self) -> None:
        """
        Run the full process.
        """
        self.extract_frames()
        predictions = self.predict_tags()
        self.aggregate_results(predictions)
        self.cleanup()


if __name__ == "__main__":
    # python src/scripts/nested_on_video.py --model_folder "/path/to/model" --video_filename "/path/to/video.mp4"
    args = get_args()
    Processor(
        model_folder=args.model_folder,
        video_filename=args.video_filename,
    ).run()