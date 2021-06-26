from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

def get_logger():
    kwargs = {'entity': 'ml43d'}
    project = "3dpart-simclr"
    id = get_run_id()
    wandb_logger = WandbLogger(project=project, name=id, id=id, **kwargs)
    return wandb_logger

def get_run_id():
    return datetime.now().strftime('%y%m%d%H%M%S%f')[:-4]
