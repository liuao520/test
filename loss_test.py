import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple

from core.hrnet import HRNet
from core.loss import JointsMSELoss
from core.make_dataset import CocoDataset
from core.make_ground_truth import GroundTruth
from core.metric import PCK
from test import test_during_training
from utils.work_flow import get_model, print_model_summary
from configuration.base_config import Config
from utils.tools import get_config_params

from utils.work_flow import get_max_preds


@flow.global_function(type="train")
def test_job(
    images: tp.Numpy.Placeholder((10, 256, 256, 3), dtype=flow.float),
    target: tp.Numpy.Placeholder((10, 64, 64, 17), dtype=flow.float),
    target_weight: tp.Numpy.Placeholder((10, 17, 1), dtype=flow.float),
    # y_pred: tp.Numpy.Placeholder((10,64,64,17), dtype=flow.float),
    ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
    loss = JointsMSELoss()

    print("!!!!!!")
    print(y_pred.shape)
    loss_metric = loss.call(y_pred, target, target_weight)
    
    loss_metric = flow.math.reduce_mean(loss_metric)
    # Set learning rate as 0.001
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    # Set Adam optimizer
    # print(loss_metric.shape)
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss_metric)

    print("-----------------")
    print(loss_metric, y_pred, target)
    return loss_metric, y_pred, target


def metric(images,target,target_weight,y_pred):
  loss_metric, y_pred, target = test_job(images,target,target_weight)
  pck = PCK()
  _, accuracy_metric, _, _ = pck.call(network_output=y_pred, target=target)
  accurary = np.mean(accuracy_metric)
  print("--@@-")
  print(loss_metric)
  print("--!-")
  # print(y_pred)
  print("----%------")
  print(target)
  print("--!-")

if __name__ == "__main__":
    # cfg = get_config_params(Config.TRAINING_CONFIG_NAME)
    # hrnet = get_model(cfg)
    #print_model_summary(hrnet)
    
    images = np.random.uniform(-10, 10, (10, 256, 256, 3)).astype(np.float32)
    target = np.random.uniform(-10, 10, (10, 64, 64, 17)).astype(np.float32)
    target_weight = np.random.uniform(-10, 10, (10, 17, 1)).astype(np.float32)
    y_pred = np.random.uniform(-10, 10, (10,64,64,17)).astype(np.float32)
    metric(images,target,target_weight,y_pred)
    
