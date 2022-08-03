import json
import logging
import sys
from pathlib import Path
from typing import Union, List, Tuple
from urllib.request import urlopen

import PIL.Image
import cv2
import numpy as np
import torch
from superai.meta_ai import BaseModel
from superai.meta_ai.base.base_ai import default_random_seed
from superai.meta_ai.parameters import HyperParameterSpec, ModelParameters
from superai.meta_ai.schema import TrainerOutput, TaskInput
from tqdm import tqdm

unet_path = Path(__file__).parent.absolute().joinpath("Pytorch-UNet")
print(unet_path)
sys.path.insert(0, str(unet_path))
try:
    import train
    import predict
    from unet import UNet
except:
    raise Exception(f"train/predict file could not be imported from {unet_path}")

logger = logging.getLogger(__name__)


class ImportedUNetModel(BaseModel):
    def load_weights(self, weights_path: str):
        pass

    def predict(self, inputs: Union[TaskInput, List[dict]], context=None):
        pass

    @staticmethod
    def extract_superai_dataset(training_data: str) -> Tuple[Path, Path]:
        npz_path = Path(training_data).joinpath("dataset_signed.npz")
        assert npz_path.exists(), f"signed dataset not found at {npz_path}"
        npz_obj = np.load(str(npz_path), allow_pickle=True)

        img_path = unet_path / "data/imgs"
        masks_path = unet_path / "data/masks"
        img_path.mkdir(parents=True, exist_ok=True)
        masks_path.mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(tqdm(npz_obj["y_train"])):
            if isinstance(entry, str):
                entry = json.loads(entry)
            json_obj = entry[0]
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        net = UNet(
            n_channels=model_parameters.get("n_channels", 3),
            n_classes=model_parameters.get("n_classes", 2),
            bilinear=model_parameters.get("bilinear", False),
        )
        if weights_path is not None and model_parameters.get("load", False) is True:
            weights_file = Path(weights_path).joinpath("weights.pth")
            net.load_state_dict(torch.load(weights_file, map_location=device))
            logger.info(f"Model weights loaded from {weights_file}")
        net.to(device)

        dir_img, dir_mask = self.extract_superai_dataset(training_data)
        train.dir_img = dir_img
        train.dir_mask = dir_mask
        train.dir_checkpoint = Path(model_save_path)

        train_method = train.train_net(
            net=net,
            epochs=hyperparameters.epochs,
            batch_size=hyperparameters.batch_size,
            learning_rate=hyperparameters.learning_rate,
            device=device,
            img_scale=hyperparameters.get("scale", 0.5),
            val_percent=hyperparameters.get("val_percent", 0.1),
            amp=hyperparameters.get("amp", False),
        )
