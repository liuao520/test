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

def __calculate_distance(pred, target, normalize):
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        distance = np.zeros((pred.shape[-1], pred.shape[0]))
        for n in range(pred.shape[0]):
            for c in range(pred.shape[-1]):
                if target[n, 0, c] > 1 and target[n, 1, c] > 1:
                    normed_preds = pred[n, :, c] / normalize[n]
                    normed_targets = target[n, :, c] / normalize[n]
                    distance[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    distance[c, n] = -1
        return distance

def __distance_accuracy(self, distance):
    distance_calculated = np.not_equal(distance, -1)
    num_dist_cal = distance_calculated.sum()
    if num_dist_cal > 0:
        return np.less(distance[distance_calculated], 0.5).sum() * 1.0 / num_dist_cal
    else:
        return -1

# @flow.global_function()
def pck(network_output,target):
    _, h, w, c = network_output.shape
    index = list(range(c))
    pred, _ = get_max_preds(heatmap_tensor=network_output)
    target, _ = get_max_preds(heatmap_tensor=target)
    normalize = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    distance = self.__calculate_distance(pred, target, normalize)

    accuracy = np.zeros((len(index) + 1))
    average_accuracy = 0
    count = 0

    for i in range(c):
        accuracy[i + 1] = self.__distance_accuracy(distance[index[i]])
        if accuracy[i + 1] > 0:
            average_accuracy += accuracy[i + 1]
            count += 1
    average_accuracy = average_accuracy / count if count != 0 else 0
    if count != 0:
        accuracy[0] = average_accuracy
    return accuracy, average_accuracy, count, pred



@flow.global_function()
def test_job(
    images: tp.Numpy.Placeholder((10, 256, 256, 3), dtype=flow.float),
    target: tp.Numpy.Placeholder((10, 64, 64, 17), dtype=flow.float),
    target_weight: tp.Numpy.Placeholder((10, 17, 1), dtype=flow.float),
    # y_pred: tp.Numpy.Placeholder((10,64,64,17), dtype=flow.float),
    ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
    loss = JointsMSELoss()
    pck = PCK()
    # gt = GroundTruth(cfg, batch_data)
    # images, target, target_weight = gt.get_ground_truth()
    # with tf.GradientTape() as tape:
    y_pred = HRNet(images, training=True)
    print("!!!!!!")
    print(y_pred.shape)
    loss_metric = loss.call(y_pred, target, target_weight)
    
    loss_metric = flow.math.reduce_mean(loss_metric)
    # Set learning rate as 0.001
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    # Set Adam optimizer
    # print(loss_metric.shape)
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss_metric)

    # flow.optimizer.Adam(lr_scheduler=0.001).minimize(loss_metric)
    # flow.losses.add_loss(loss_metric)
    # flow.optimizer.Adam(lr_scheduler=1e-3).minimize(loss_metric)
    print("-----------------")
    print(loss_metric, y_pred, target)
    return loss_metric, y_pred, target
    # _, accuracy_metric, _, _ = pck.call(network_output=y_pred, target=target)
    # accuracy_metric = flow.math.reduce_mean(accuracy_metric)
    # print(loss_metric,accuracy_metric)
    # # flow.losses.add_loss(accuracy_metric)
    # # print("Epoch: {}/{}, step: {}/{}, loss: {:.10f}, accuracy: {:.5f}".format(epoch,
    # #                       cfg.EPOCHS,
    # #                       step,
    # #                       flow.math.ceil(dataset_length / cfg.BATCH_SIZE),
    # #                       loss_metric,
    # #                       accuracy_metric))
    # flow.optimizer.Adam(lr_scheduler=1e-3).minimize(loss_metric)


    # return outputs

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
    outputs = metric(images,target,target_weight,y_pred)
    print(outputs)