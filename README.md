# ai-unet
SuperAI BaseModel compatible implementation of UNet from https://github.com/milesial/Pytorch-UNet

This implementation illustrates how an open source model can be converted into a super.ai BaseModel compatible implementation which can be run on our backend. 

## Training method
The Pytorch-UNet implementation contains simple train script containing methods which start training, namely the `train_net` method. This initializes the model, sets up the data loader and performs metric tracking based on the hyperparameters passed in the arguments. We plan on re-using this method to prevent code duplication. 

> We containerize the code for better deployment, so the following instructions are containerization focused.

1. Enable importing

You can import the implementation in a deployed container by cloning the repository in the container. This can be done by creating a script like [`setup.sh`](setup/setup.sh). This script will be added to the deployment by adding it to your config as shown below (check [config](config) folder)
```yaml
template:
  name: "UNetTemplate"
  description: "Template for UNet based on https://github.com/milesial/Pytorch-UNet"
  model_class: "ImportedUNetModel"
  artifacts:
    run: "setup/setup.sh"
  conda_env: "conda.yml"
...
```
This places the `Pytorch-UNet` code on your home directory of the container. 

3. Installing requirements:

The next step is installing all requirements. We have specified a [`conda.yml`](conda.yml) file. You could specify the installation instructions in the setup script above. 

4. Import the modules:

To make sure you import all modules correctly, add it to sys path
```python
from pathlib import Path
import sys
from superai.meta_ai import BaseModel

unet_path = Path(__file__).parent.absolute().joinpath("Pytorch-UNet")
sys.path.insert(0, str(unet_path))
try:
    import train
    import predict
    from unet import UNet
except:
    raise Exception(f"train/predict file could not be imported from {unet_path}")

class ImportedUNetModel(BaseModel):
...
```

> ### Points to note while training
> 
> To ensure that the training works with a superai app, there is some preprocessing required to obtain the data from the superai datasets. Please see the `train` implementation in [`ImportedUNetModel.py`](ImportedUNetModel.py) for more details
> 
> Currently, the metrics are generated in wandb in a anonymous run. This will be changed in the future.

### Testing

If you have a local environment setup will all requirements present, you can run training locally. This can be done by running 
```bash
superai ai method train -p .AISave/unet/1/ -tp superai_dataset -mp gen_model -h batch_size=1
```
For generating content of the `.AISave` folder, run the remote deployment script
```bash
superai ai training deploy --config-file config/dev_config.yaml
```

## Predict

If all the requirements are satisfied, you can deploy a predictor locally by running
```bash
superai ai deploy --config-file config/dev_config.yaml
```
Easiest way to check logs is by checking the docker desktop and looking for a container named `unet:1`

To send prediction requests to this model
```bash
superai ai predictor-test -i '<JSON input>' --config-file config/dev_config.yaml
```

You can update the orchestrator in the config file to `AWS_EKS` to deploy on the clous and repeat the above experiment. 