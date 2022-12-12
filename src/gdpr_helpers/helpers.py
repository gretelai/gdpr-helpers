from pprint import pprint
from time import sleep

from tqdm import tqdm


def quiet_poll(model):

    running = True
    no_epochs = True
    epoch = 0
    index = 0
    last_status = ""

    with tqdm() as pbar:
        while running:
            model._poll_job_endpoint()
            pbar.update(1)
            sleep(1)

            # Account for different ways of logging job status
            data_dict = model.__dict__["_data"]
            if "status" in data_dict.keys():
                status = data_dict["status"]
                model_type = data_dict["model_type"]
            elif "model" in data_dict.keys():
                status = data_dict["model"]["status"]
                model_type = data_dict["model"]["model_type"]
            elif "handler" in data_dict.keys():
                status = data_dict["handler"]["status"]
                model_type = data_dict["handler"]["model_type"]
            logs = data_dict["logs"]
            status_msg = f"{model_type.capitalize()} - Job {status}"

            # Add detailed status when available
            if logs:
                if "ctx" in logs[-1].keys():
                    if logs[-1]["ctx"]:
                        pbar.set_postfix(logs[-1]["ctx"])
                    else:
                        status_msg = (
                            logs[-1]["msg"] if len(logs[-1]["msg"]) < 25 else status_msg
                        )

            pbar.set_description(status_msg)

            if status == "completed" or status == "error":
                running = False

            if status != last_status:
                last_status = status
