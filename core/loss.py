# import tensorflow as tf
import oneflow as flow
import numpy as np

class JointsMSELoss(object):
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def call(self, y_pred, target, target_weight):
        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]
        print(batch_size,num_of_joints)
        print('---')
        pred = flow.reshape(x=y_pred, shape=(batch_size, -1, num_of_joints))
        '''#注意一下,,这里的格式可能会出现问题'''
        # heatmap_pred_list = np.split(pred, num_of_joints, axis=2)
        heatmap_pred_list = []
        for i in range(num_of_joints):
          tensor = flow.slice(pred, begin=[None, None, i*1], size=[None, None, 1])
          heatmap_pred_list.append(tensor)
        
        # heatmap_pred_list = pred.with_distribue(distribute.split(1))
        gt = flow.reshape(x=target, shape=(batch_size, -1, num_of_joints))
        # heatmap_gt_list = np.split(gt, num_of_joints, axis=2)
        heatmap_gt_list = []
        for i in range(num_of_joints):
          tensor = flow.slice(gt, begin=[None, None, i*1], size=[None, None, 1])
          heatmap_gt_list.append(tensor)
        # heatmap_gt_list = gt.with_distribue(distribute.split(1))
        print(target_weight.shape)
        loss = 0.0
        for i in range(num_of_joints):
            heatmap_pred = flow.squeeze(heatmap_pred_list[i])
            heatmap_gt = flow.squeeze(heatmap_gt_list[i])
            # print(heatmap_pred.shape)
            # print(heatmap_gt.shape)
            # flow.reshape(flow.slice(target_weight, begin=[None,i*1, None], size=[None,1,None]),[10,1])
            # temp = flow.math.square(heatmap_pred * target_weight[:, i] - heatmap_gt * target_weight[:, i])
            temp = flow.math.square(heatmap_pred * flow.reshape(flow.slice(target_weight, begin=[None,i*1, None], size=[None,1,None]),[10,1]) - heatmap_gt * flow.reshape(flow.slice(target_weight, begin=[None,i*1, None], size=[None,1,None]),[10,1]))
            loss += 0.5 * flow.math.reduce_mean(temp, axis=1, keepdims=True)
            # loss += 0.5 * mse_(y_true=heatmap_pred * target_weight[:, i],
            #                        y_pred=heatmap_gt * target_weight[:, i])
        return loss / num_of_joints

    # @staticmethod
    # def mse_(x, y):
    #     temp = flow.math.square(x - y)
    #     mse = flow.math.reduce_mean(temp, axis=1, keepdims=True)
    #     return mse
