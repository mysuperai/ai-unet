# Test Suite

There are a couple of tests added here which check the functionality of the code. For the `upload_url` feature, we are going to use localstack. 
The instructions for running the tests are as follows:

### Install requirements
**IMPORTANT:** Using the localstack fixtures requires python 3.10 as there is some specific syntax in the implementation for that python version, which is not backwards compatible with older versions. So your local venv could be based on 3.10
```bash
pip install -r test_requirements.txt
```

### Start localstack service, run tests and stop
```bash
localstack start -d
pytest .
localstack stop
```

> Later the localstack service will be moved into a fixture.