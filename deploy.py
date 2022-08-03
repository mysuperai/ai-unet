import shutil
from pathlib import Path

from superai.meta_ai import AITemplate, AI
from superai.meta_ai.image_builder import Orchestrator
from superai.meta_ai.parameters import Config
from superai.meta_ai.schema import Schema

if Path(".AISave").exists():
    shutil.rmtree(".AISave")


template = AITemplate(
    input_schema=Schema(),
    output_schema=Schema(),
    configuration=Config(),
    name="UNet_Template",
    description="Template for UNet model",
    model_class="ImportedUNetModel",
    requirements=[
        "matplotlib",
        "numpy",
        "Pillow",
        "torch<1.11.0",
        "torchvision",
        "tqdm",
        "wandb",
        "opencv-python-headless==4.5.5.64",
    ],
    artifacts={"run": "setup/setup.sh"},
)

ai = AI(
    ai_template=template,
    input_params=template.input_schema.parameters(),
    output_params=template.output_schema.parameters(),
    name="unet",
    version=1,
    description="Unet AI instance",
    weights_path="gen_model",
)

predictor = ai.deploy(
    orchestrator=Orchestrator.AWS_EKS, enable_cuda=True
)
