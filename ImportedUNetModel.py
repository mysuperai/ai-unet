import json
import logging
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Tuple
from urllib.request import urlopen, urlretrieve

import PIL.Image
import cv2
import numpy as np
import requests
import superai_schema.universal_schema.task_schema_functions as df
import torch
from superai.meta_ai import BaseModel
from superai.meta_ai.base.base_ai import default_random_seed
from superai.meta_ai.parameters import HyperParameterSpec, ModelParameters
from superai.meta_ai.schema import TrainerOutput
from superai.utils import retry
from tqdm import tqdm

from superai_helper import get_image_url, obtain_schema_bound_results

# add the `Pytorch-UNet` implementation in the sys path to import the modules correctly.
unet_path = Path(__file__).parent.absolute().joinpath("Pytorch-UNet")
sys.path.insert(0, str(unet_path))
try:
    import train
    import predict
    from unet import UNet
except:
    raise Exception(f"train/predict file could not be imported from {unet_path}")

logger = logging.getLogger(__name__)


class ImportedUNetModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # default configuration, can be overridden by config generated during training
        self.config = {
            "n_channels": 3,
            "n_classes": 2,
            "bilinear": False,
            "scale": 0.5,
            "class_name": "Car",
        }
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device {self.device}")

    def load_weights(self, weights_path: str):
        """
        Run during prediction, this method loads the weights into a model instance and makes it available to the prediction service.
        """
        logger.info(f"Loading model {weights_path}")

        # read config if it exists.
        if Path(weights_path).joinpath("config.json").exists():
            with open(Path(weights_path).joinpath("config.json"), "r") as config_json:
                self.config = json.load(config_json)

        # initialize model
        self.net = UNet(
            n_channels=self.config["n_channels"],
            n_classes=self.config["n_classes"],
            bilinear=self.config["bilinear"],
        )
        self.net.to(self.device)

        # find weights file matching a pattern
        model_path = sorted(Path(weights_path).glob("checkpoint*.pth"), reverse=True)
        if len(model_path) == 0:
            raise FileNotFoundError(
                f"No checkpoint found in {weights_path}. Content: {list(Path(weights_path).iterdir())}"
            )

        # load weights
        self.net.load_state_dict(torch.load(Path(weights_path).joinpath(model_path[0])))
        logger.info("Model loaded")

    @staticmethod
    def save_mask_as_file(mask: object, inst: int, predict_dir: str):
        """
        Saves the mask file locally before preparing them for upload
        Format should be of the form .png or .jpg or empty
        """
        im = PIL.Image.fromarray(mask)
        im_filename = f"mask{inst}.png"
        im.save(f"{predict_dir}/{im_filename}")
        return im_filename

    def _handle_mask(self, prediction, inst, prediction_dir):
        new_masks = prediction["new_masks"]
        scores = prediction["scores"]

        logger.info(f"processing mask number {inst}")
        filename = self.save_mask_as_file(new_masks[..., inst], inst, prediction_dir)
        data_uri = f"data://{filename}"

        schema_object = df.image_segment(
            mask_url=data_uri,
            selection=df.choice(value=self.config["class_name"], tag="0")[
                "schema_instance"
            ],
            color="#FFFFFF",
            index=inst,
        )["schema_instance"]
        return {"prediction": schema_object, "score": float(scores[inst])}

    @staticmethod
    def tar_output_files(prediction_dir):
        """
        Create a tar for prediction output. Useful if there are more than one mask
        :param prediction_dir: Directory containing all predictions
        :return: Output file
        """
        file_out = BytesIO()
        with tarfile.open(mode="w:gz", fileobj=file_out) as archive:
            archive.add(prediction_dir, arcname="prediction", recursive=True)
            logger.info("Created tarfile for prediction output")
        return file_out

    @retry(Exception, tries=5, delay=0.5, backoff=1)
    def upload_tarobj(self, tarobj, upload_url):
        """
        Helper to upload the tarobj to the upload_url.
        """
        headers = {"Content-Type": "application/gzip"}
        response = requests.put(upload_url, data=tarobj.getbuffer(), headers=headers)
        if response.status_code != 200:
            logger.info(response.text)
            raise Exception("Mask Upload failed.")
        logger.info(
            f"Upload of tarfile completed with status code {response.status_code}"
        )

    def predict(self, task_inputs, context=None):
        """
        Predict method to be overridden in the superai BaseModel. The implementation here depends on `load_weights`
         method to load the model. This function runs as a service where we send json requests containing task_inputs
        :param task_inputs: JSON containing the task inputs.
        :return: JSON result in the form of {"prediction": <JSON object>, "score": <float>}
        """
        logger.info(f"Task inputs: {task_inputs}")

        # obtain the download and upload URL for the image and mask respectively.
        upload_url = task_inputs["upload_url"]
        image_url, task_inputs = get_image_url(task_inputs)
        # download image
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            urlretrieve(image_url, temp_file.name)
            logger.info("image retrieved")

            # predict from model
            prediction = self.predict_from_image(temp_file.name)

        logger.info("predict from image done.")

        instances = []
        with tempfile.TemporaryDirectory() as prediction_dir:
            # store masks in local temp file system
            for inst in range(len(prediction["new_masks"])):
                instances.append(self._handle_mask(prediction, inst, prediction_dir))

            # compress and upload the masks
            logger.info("everything done, uploading data.")
            tar_obj = self.tar_output_files(prediction_dir)
            self.upload_tarobj(tar_obj, upload_url)

        # generate JSON output
        result = obtain_schema_bound_results(instances, task_inputs)

        logger.info(f"Prediction result {result}")
        return result

    def predict_from_image(self, path):
        """
        The main predict method calling the `predict_img` method from the Pytorch_UNet implementation.
        :param path: Path to input file
        :return: dictionary of masks and scores. Here the scoring can be fixed.
        """
        image = PIL.Image.open(path)
        mask = predict.predict_img(
            net=self.net,
            full_img=image,
            scale_factor=self.config.get("scale", 0.5),
            out_threshold=self.config.get("mask_threshold", 0.5),
            device=self.device,
        )

        return {"new_masks": [mask], "scores": [1.0]}

    @staticmethod
    def extract_superai_dataset(training_data: str) -> Tuple[Path, Path]:
        """
        Helper to extract out the training data image URLs from the superai AI dataset present in the `training_data`
        dir
        In this function, we are basically replicating the dataset preparation done in `Pytorch_UNet` implementation by
         downloading the files in the same directories as required by the implementation

            Pytorch_UNet
            ...
            |-data
                |-imgs
                |-masks
            ...

        We pass these paths back to the training module to load the dataset correctly from the right paths.
        """
        # Obtain the path to file containing signed urls
        npz_path = Path(training_data).joinpath("dataset_signed.npz")
        assert npz_path.exists(), f"signed dataset not found at {npz_path}"
        npz_obj = np.load(str(npz_path), allow_pickle=True)

        # prepare folders where the images are to be downloaded
        img_path = unet_path / "data/imgs"
        masks_path = unet_path / "data/masks"
        img_path.mkdir(parents=True, exist_ok=True)
        masks_path.mkdir(parents=True, exist_ok=True)

        # Download images
        for i, entry in enumerate(tqdm(npz_obj["y_train"])):
            if isinstance(entry, str):
                entry = json.loads(entry)
            json_obj = entry[0]

            # provide a unified filename to identify the image and its
            # corresponding mask (only one in this implementation)
            filename = f"{i:07d}"
            img_url = json_obj["schema_instance"]["imageUrl"]
            mask_url = json_obj["schema_instance"]["annotations"]["instances"][0][
                "maskUrl"
            ]
            with urlopen(img_url) as request:
                img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = PIL.Image.fromarray(img)
                img.save(img_path / f"{filename}.jpg", optimize=True)
            with urlopen(mask_url) as request:
                img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                img = PIL.Image.fromarray(img)
                img.save(masks_path / f"{filename}_mask.gif", bits=1, optimize=True)

        return img_path, masks_path

    def train(
        self,
        model_save_path,
        training_data,
        validation_data=None,
        test_data=None,
        production_data=None,
        weights_path=None,
        encoder_trainable: bool = True,
        decoder_trainable: bool = True,
        hyperparameters: HyperParameterSpec = None,
        model_parameters: ModelParameters = None,
        callbacks=None,
        random_seed=default_random_seed,
    ) -> TrainerOutput:
        """
        Train method to be overriden. The contents of this method are run as a job when a training run is created.
        Essentially, here we are downloading the data obtained from the app dataset, and calling the train method
        present in `Pytorch_UNet` implementation. This removes necessity of any code duplication.

        Currently, all metrics are written to an anonymous run in wandb. Later this could be refactored to be written
        into superai's internal tracking methodology.
        """
        # initialize model
        net = UNet(
            n_channels=model_parameters.get("n_channels", 3),
            n_classes=model_parameters.get("n_classes", 2),
            bilinear=model_parameters.get("bilinear", False),
        )
        if weights_path is not None and model_parameters.get("load", False) is True:
            # if weight loading is required
            weights_file = Path(weights_path).joinpath("weights.pth")
            net.load_state_dict(torch.load(weights_file, map_location=self.device))
            logger.info(f"Model weights loaded from {weights_file}")
        net.to(self.device)

        # download and extract images from the app dataset
        dir_img, dir_mask = self.extract_superai_dataset(training_data)
        train.dir_img = dir_img
        train.dir_mask = dir_mask
        train.dir_checkpoint = Path(model_save_path)

        # train using the train_net method from `Pytorch_UNet` implementation
        train_method = train.train_net(
            net=net,
            epochs=hyperparameters.epochs,
            batch_size=hyperparameters.batch_size,
            learning_rate=hyperparameters.learning_rate,
            device=self.device,
            img_scale=hyperparameters.get("scale", 0.5),
            val_percent=hyperparameters.get("val_percent", 0.1),
            amp=hyperparameters.get("amp", False),
        )

        # save some essential config which will be useful while loading the prediction model.
        config = {
            "n_channels": model_parameters.get("n_channels", 3),
            "n_classes": model_parameters.get("n_classes", 2),
            "bilinear": model_parameters.get("bilinear", False),
            "scale": hyperparameters.get("scale", 0.5),
            "class_name": "Car",  # can be retrieved from the npz files
        }
        with open(Path(model_save_path) / "config.json", "w") as config_json:
            json.dump(config, config_json)
