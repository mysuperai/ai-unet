"""
Contains helper functions which will be moved into the superai SDK in the future
"""

import logging
import re

logger = logging.getLogger(__name__)


def check_schema_type(task_inputs):
    if "schema_instance" in task_inputs["parameters"]["output_schema"]:
        return True
    else:
        logger.fatal(
            "Can't find neither old nor new schema paradigm, can't return a result"
        )
        raise ValueError(
            "Can't find neither old nor new schema paradigm, can't return a result"
        )


def get_image_url(task_inputs):
    check_schema_type(task_inputs)
    image_url = task_inputs["parameters"]["output_schema"]["schema_instance"][
        "imageUrl"
    ]
    task_inputs["parameters"]["output_schema"]["schema_instance"][
        "imageUrl"
    ] = convert_to_dataurl(image_url)
    return image_url, task_inputs


def obtain_schema_bound_results(instances, task_inputs):
    empty_prediction = False
    if not instances:
        empty_prediction = True
        instances = [{"prediction": {}}]
    task_inputs["parameters"]["output_schema"]["schema_instance"]["annotations"] = {
        "instances": [instance["prediction"] for instance in instances]
    }
    output_prediction = [task_inputs["parameters"]["output_schema"]]

    result = {
        "prediction": output_prediction,
        "score": (sum(instance["score"] for instance in instances) / len(instances))
        if not empty_prediction
        else 0,
    }
    return result


def convert_to_dataurl(url):
    for cloudfront in [
        "d14qrv7r39xn2f.cloudfront.net",  # dev
        "d2cpodczaz39bz.cloudfront.net",  # prod
        "drf9wm7h7lj0m.cloudfront.net",  # sandbox
        "d3kxzxf0vuui76.cloudfront.net",  # stg
    ]:
        if cloudfront in url:
            match = re.search(r"(?<=https:)(.*)(?=\?)", url).group(0)
            substring = "data:" + match.replace(cloudfront + "/", "")
            logger.info(
                f"Signed URL found. Returning this data URL instead -> {substring}"
            )
            return substring
    return url
