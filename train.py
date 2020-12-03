# import tensorflow as tf
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import math

from core.loss import JointsMSELoss
from core.make_dataset import CocoDataset
from core.make_ground_truth import GroundTruth
from core.metric import PCK
from test import test_during_training
from core.hrnet import HRNet
from utils.work_flow import get_model, print_model_summary
from configuration.base_config import Config
from utils.tools import get_config_params


cfg = get_config_params(Config.TRAINING_CONFIG_NAME)

if __name__ == '__main__':
    # # GPU settings
    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    cfg = get_config_params(Config.TRAINING_CONFIG_NAME)
    # hrnet = get_model(cfg)
    # print_model_summary(hrnet)

    # Dataset
    coco = CocoDataset(config_params=cfg, dataset_type="train")
    dataset, dataset_length = coco.generate_dataset()

    # loss and optimizer
    loss = JointsMSELoss()
    # optimizer = flow.optimizer.Adam(lr_scheduler=1e-3)
    # metircs
    # loss_metric = tf.metrics.Mean()
    pck = PCK()
    # accuracy_metric = tf.metrics.Mean()
    
    @flow.global_function(type="train")
    def train_step(images: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 256, 256, 3), dtype=flow.float32),
            target: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 64, 64, 17), dtype=flow.float32),
            target_weight: tp.Numpy.Placeholder((cfg.BATCH_SIZE, 17, 1), dtype=flow.float32),
            # y_pred: tp.Numpy.Placeholder((10,64,64,17), dtype=flow.float),
            ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy]:
        
        y_pred = HRNet(images, training=True)
        loss_metric = loss.call(y_pred, target, target_weight)
        loss_metric = flow.math.reduce_mean(loss_metric)
        # Set learning rate as 0.001
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
        # Set Adam optimizer
        # print(loss_metric.shape)
        flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss_metric)
        return loss_metric, y_pred, target
        

    check_point = flow.train.CheckPoint()
    start_epoch = cfg.LOAD_WEIGHTS_FROM_EPOCH

    if cfg.LOAD_WEIGHTS_BEFORE_TRAINING:
        # hrnet.load_weights(filepath=cfg.save_weights_dir + "epoch-{}".format(start_epoch))
        check_point.load(cfg.save_weights_dir + "epoch-{}".format(start_epoch))
        print("Successfully load weights!")
    else:
        start_epoch = -1

    # check_point.init()
    for epoch in range(start_epoch + 1, cfg.EPOCHS):
        for step, batch_data in enumerate(dataset):
            gt = GroundTruth(cfg, batch_data)
            images, target, target_weight = gt.get_ground_truth()
            print(images.dtype, target.dtype, target_weight.dtype)
            print(images.shape, target.shape, target_weight.shape)
            loss_metric, y_pred, target = train_step(images, target, target_weight)

            _, accuracy_metric, _, _ = pck.call(network_output=y_pred, target=target)
            accurary = np.mean(accuracy_metric)

           
            print("Epoch: {}/{}, step: {}/{}, loss: {:.10f}, accuracy: {:.5f}".format(epoch,
                                                    cfg.EPOCHS,
                                                    step,
                                                    math.ceil(dataset_length / cfg.BATCH_SIZE),
                                                    loss_metric,
                                                    accurary))
            

        if epoch % cfg.SAVE_FREQUENCY == 0:
            check_point.save(cfg.save_weights_dir + "epoch-{}".format(epoch))
        if cfg.TEST_DURING_TRAINING:
            test_during_training(cfg=cfg, epoch=epoch, model=hrnet)


    check_point.save(cfg.save_weights_dir+"saved_model")

