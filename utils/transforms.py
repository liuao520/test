# import tensorflow as tf
import oneflow as flow
import cv2

from PIL import Image
import numpy as np
import oneflow.typing as tp
from typing import Tuple

from PIL import Image


def read_image(image_dir, cfg):

    image= Image.open(image_dir)
    # image_files = [open(im, "rb") for im in images]image.shape
    # image = image.read()#[imf.read() for imf in image_files]\
    # image=np.array(image).astype(np.float32)
    # print(image.shape)
    
    return image

    # # image_content = tf.io.read_file(filename=image_dir)
    # img=Image.open(image_dir)
    # #转换成np.ndarray格式
    # image=np.array(img)
    # print("img2:",image.shape)
    # print("img2:",type(image))
    # print("-"*10)

    # image_content = _of_image_decode(image_dir)
    # The 'image' has been normalized.  这个到底是什么,感觉会有问题  要解码为tnsor吗
    # image = tf.io.decode_image(contents=image_content, channels=cfg.CHANNELS, dtype=tf.dtypes.float32)
    # image = flow.image.decode(images_bytes_buffer=image_content, dtype=flow.float32)#color_space
    # return image


# Determine whether a point is within a rectangular border.
def point_in_rect(point_x, point_y, rect):
    # rect : (x, y, w, h)
    xmin = rect[0]
    ymin = rect[1]
    xmax = xmin + rect[2]
    ymax = ymin + rect[3]
    if xmin <= point_x <= xmax and ymin <= point_y <= ymax:
        is_point_in_rect = True
    else:
        is_point_in_rect = False
    return is_point_in_rect


class RandomCropTransform(object):
    def __init__(self, image, keypoints, bbox, resize_h, resize_w, num_of_joints):
        self.image = image
        self.keypoints = keypoints
        self.bbox = bbox
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.num_of_joints = num_of_joints

    @flow.global_function()
    def image_transform(self):
        if self.resize_h != self.resize_w:
            raise ValueError("The values of resize_h and resize_w should be equal.")
        # human_instance = tf.image.crop_to_bounding_box(image=self.image,
                    # offset_height=self.bbox[1],
                    # offset_width=self.bbox[0],
                    # target_height=self.bbox[3],
                    # target_width=self.bbox[2])
        # human_instance = flow.image.crop_mirror_normalize(input_blob=self.image,
        #                           crop_h=self.bbox[3],
        #                           crop_w=self.bbox[2],
        #                           crop_pos_y=self.bbox[1],
        #                           crop_pos_x=self.bbox[0])
        human_instance = self.image.crop((self.bbox[0],self.bbox[1],self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]))
        human_instance_size = np.array(human_instance).astype(np.float32)
        left_top_of_human_instance = self.bbox[0:2]
        crop_rect, cropped_image = self.__random_crop_in_roi(image=self.image, roi=human_instance_size,
                                                             left_top_of_roi=left_top_of_human_instance)
        resize_ratio = self.resize_h / crop_rect.shape[-1]
        # resized_image = tf.image.resize(images=cropped_image, size=[self.resize_h, self.resize_w])
        resized_image = cropped_image.resize([self.resize_h, self.resize_w])#flow.image.Resize(images=cropped_image, target_size=[self.resize_h, self.resize_w])
        resized_image = np.array(resized_image).astype(np.float32)
        return resized_image, resize_ratio, crop_rect

    # @flow.global_function()
    def keypoints_transform(self, resize_ratio, crop_rect):
        crop_rect = crop_rect.numpy()
        transformed_keypoints = self.keypoints.numpy()
        # First determine whether the point is inside the crop area.
        for i in range(self.num_of_joints):
            if not point_in_rect(point_x=transformed_keypoints[i, 0], point_y=transformed_keypoints[i, 1], rect=crop_rect):
                transformed_keypoints[i, 2] = 0.0

        for i in range(self.num_of_joints):
            if transformed_keypoints[i, 2] > 0.0:
                # Calculate the coordinates of the keypoints after cropping the original picture.
                transformed_keypoints[i, 0] = transformed_keypoints[i, 0] - crop_rect[0]
                transformed_keypoints[i, 1] = transformed_keypoints[i, 1] - crop_rect[1]
                # Calculate the coordinates of the keypoints after resizing.
                transformed_keypoints[i, 0] = int(transformed_keypoints[i, 0] * resize_ratio)
                transformed_keypoints[i, 1] = int(transformed_keypoints[i, 1] * resize_ratio)
        return transformed_keypoints

    def __random_crop_in_roi(self, image, roi, left_top_of_roi):
        roi_h = roi.shape[0]
        roi_w = roi.shape[1]
        if roi_h > roi_w:
            longer_border = roi_h
            shorter_border = roi_w
        else:
            longer_border = roi_w
            shorter_border = roi_h
        random_coord = flow.random_uniform_initializer(minval=0, maxval=longer_border - shorter_border)
        if longer_border == roi_h:
            x_random_crop = left_top_of_roi[0]
            y_random_crop = int(left_top_of_roi[1] + random_coord)
        else:
            x_random_crop = int(left_top_of_roi[0] + random_coord)
            y_random_crop = left_top_of_roi[1]
        # crop_rect = tf.convert_to_tensor(value=[x_random_crop, y_random_crop, shorter_border, shorter_border],
        #                                  dtype=tf.dtypes.int32)
        crop_rect = np.array(x_random_crop, y_random_crop, shorter_border, shorter_border).astype(np.int32)
        # cropped_image = tf.image.crop_to_bounding_box(image=image,
#                                               offset_height=y_random_crop,
#                                               offset_width=x_random_crop,
#                                               target_height=shorter_border,
#                                               target_width=shorter_border)
        
        cropped_image = image.crop((shorter_border,shorter_border,shorter_border+x_random_crop,shorter_border+y_random_crop))
        # cropped_image = flow.image.crop_mirror_normalize(input_blob=image,
        #                         crop_h=shorter_border,
        #                         crop_w=shorter_border,
        #                         crop_pos_y=y_random_crop,
        #                         crop_pos_x=x_random_crop)
        return crop_rect, cropped_image


class ResizeTransform(object):
    def __init__(self, image, keypoints, bbox, resize_h, resize_w, num_of_joints):
        self.image = image
        self.keypoints = keypoints
        self.bbox = bbox
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.num_of_joints = num_of_joints

    def image_transform(self):
        # human_instance = tf.image.crop_to_bounding_box(image=self.image,
                              # offset_height=self.bbox[1],
                              # offset_width=self.bbox[0],
                              # target_height=self.bbox[3],
        # print(self.bbox)          #                                    target_width=self.bbox[2])
        # human_instance = flow.image.crop_mirror_normalize(input_blob=self.image,
        #                           crop_h=self.bbox[3],
        #                           crop_w=self.bbox[2],
        #                           crop_pos_y=self.bbox[1],
        #                           crop_pos_x=self.bbox[0])
        human_instance = self.image.crop((self.bbox[0],self.bbox[1],self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]))
        human_instance_size = np.array(human_instance).astype(np.float32)
        # print(" simage:",human_instance.shape)
        left_top_of_human_instance = self.bbox[0:2]
        resize_ratio = [self.resize_h / human_instance_size.shape[0], self.resize_w / human_instance_size.shape[1]]
        # resized_image = tf.image.resize(images=human_instance, size=[self.resize_h, self.resize_w])
        resized_image = human_instance.resize([self.resize_h, self.resize_w])#flow.image.Resize(image=human_instance, target_size=[self.resize_h, self.resize_w])
        resized_image = np.array(resized_image).astype(np.float32)
        # print("image:",resized_image.shape)
        
        return resized_image, resize_ratio, left_top_of_human_instance

    def keypoints_transform(self, resize_ratio, left_top):
        transformed_keypoints = np.array(self.keypoints)
        for i in range(self.num_of_joints):
            if transformed_keypoints[i, 2] > 0.0:
                transformed_keypoints[i, 0] = int((transformed_keypoints[i, 0] - left_top[0]) * resize_ratio[1])
                transformed_keypoints[i, 1] = int((transformed_keypoints[i, 1] - left_top[1]) * resize_ratio[0])
        return transformed_keypoints


class KeypointsRescaleToOriginal(object):
    def __init__(self, input_image_height, input_image_width, heatmap_h, heatmap_w, original_image_size):
        self.scale_ratio = [input_image_height / heatmap_h, input_image_width / heatmap_w]
        self.original_scale_ratio = [original_image_size[0] / input_image_height, original_image_size[1] / input_image_width]

    def __scale_to_input_size(self, x, y):
        return x * self.scale_ratio[1], y * self.scale_ratio[0]

    def __call__(self, x, y):
        temp_x, temp_y = self.__scale_to_input_size(x=x, y=y)
        return int(temp_x * self.original_scale_ratio[1]), int(temp_y * self.original_scale_ratio[0])



