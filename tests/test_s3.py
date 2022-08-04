from pathlib import Path

import pytest
from superai.meta_ai import AITemplate, AI
from superai.meta_ai.parameters import Config
from superai.meta_ai.schema import Schema

try:
    # prevents import optimization
    from localstack.testing.pytest.fixtures import s3_resource, s3_client
except ImportError:
    raise


@pytest.fixture
def ai_object():
    template = AITemplate(
        input_schema=Schema(),
        output_schema=Schema(),
        configuration=Config(),
        name="Unet_template",
        description="template of Unet model",
        model_class="SuperaiUNetModel",
        model_class_path=str(Path(__file__).parent.parent.absolute()),
        artifacts={"run": "setup/setup.sh"},
        conda_env=str(Path(__file__).parent.parent.absolute() / "conda.yml"),
    )
    instance = AI(
        ai_template=template,
        input_params=template.input_schema.parameters(),
        output_params=template.output_schema.parameters(),
        name="unet",
        version=1,
        weights_path=str(Path(__file__).parent.parent.absolute() / "gen_model"),
    )
    yield instance


def test_s3_bucket_creation(s3_resource):
    assert len(list(s3_resource.buckets.all())) == 0
    bucket = s3_resource.Bucket("demo-bucket")
    bucket.create()
    assert len(list(s3_resource.buckets.all())) == 1
    bucket.delete()


def test_generate_upload_url(s3_client, s3_resource):
    bucket = s3_resource.Bucket("demo-bucket")
    bucket.create()
    url = s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": "demo-bucket", "Key": "file.tar.gz"},
        ExpiresIn=1000,
    )
    assert url
    bucket.delete()


@pytest.fixture()
def generate_upload_url(s3_client, s3_resource):
    bucket = s3_resource.Bucket("demo-bucket")
    bucket.create()
    url = s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": "demo-bucket", "Key": "file.tar.gz"},
        ExpiresIn=1000,
    )
    assert url
    yield url
    bucket.delete()


def test_predict(ai_object, generate_upload_url):
    assert ai_object.predict(inputs={"upload_url": generate_upload_url, })
