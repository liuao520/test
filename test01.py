import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple

from core.HrNet_ import HRNet
from core.loss import JointsMSELoss
from core.make_dataset import CocoDataset
from core.make_ground_truth import GroundTruth
from core.metric import PCK
from test import test_during_training
from utils.work_flow import get_model, print_model_summary
from configuration.base_config import Config
from utils.tools import get_config_params

@flow.global_function()
def test_job(
    images: tp.Numpy.Placeholder((1, 256, 192, 3), dtype=flow.float),
    ) -> tp.Numpy:
    outputs = HRNet(images, training=True)

    return outputs


if __name__ == "__main__":
    # cfg = get_config_params(Config.TRAINING_CONFIG_NAME)
    # hrnet = get_model(cfg)
    #print_model_summary(hrnet)
    
    images_in = np.random.uniform(-10, 10, (1, 256, 192, 3)).astype(np.float32)
    outputs = test_job(images_in)
    print(outputs)