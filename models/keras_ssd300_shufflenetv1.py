'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, SeparableConv2D, Dropout, BatchNormalization
from keras.layers import DepthwiseConv2D, AveragePooling2D, Add
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

import sys, os
import light_networks.shufflenetv1.shufflenet as shufflenet

def ssd_300(image_size,
            n_classes,
            input_tensor = None,
            mode='training',
            groups=3,
            scale_factor=1.0,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        input_tensor: Tensor with shape (batch, height, width, channels)
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################# 
    # Functions for Shufflenetv1 architeture
    #############################################################################
    
    def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1, linear=False):
        """
        creates a shuffleunit
        Parameters
        ----------
        inputs:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        strides:
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number
        block: int(1)
            block number
        Returns
        -------
        """
        if K.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        prefix = 'stage%d/block%d' % (stage, block)

        # default: 1/4 of the output channel of a ShuffleNet Unit
        bottleneck_channels = int(out_channels * bottleneck_ratio)
        groups = (1 if stage == 2 and block == 1 else groups)

        x = _group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                        groups=(1 if stage == 2 and block == 1 else groups),
                        name='%s/1x1_gconv_1' % prefix)
        x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
        x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

        x = Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
                            strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
        x = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

        x = _group_conv(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels,
                        groups=groups, name='%s/1x1_gconv_2' % prefix)
        x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

        if strides < 2:
            ret = Add(name='%s/add' % prefix)([x, inputs])
        else:
            avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
            ret = Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

        if linear:
            return Activation('relu', name='%s/relu_out' % prefix)(ret), ret

        ret = Activation('relu', name='%s/relu_out' % prefix)(ret)

        return ret

    def _group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
        """
        grouped convolution


        Parameters
        ----------
        x:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        groups:
            number of groups per channel
        kernel: int(1)
            An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        stride: int(1)
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for all spatial dimensions.
        name: str
            A string to specifies the layer name

        Returns
        -------

        """
        if groups == 1:
            return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                        use_bias=False, strides=stride, name=name)(x)

        # number of intput channels per group
        ig = in_channels // groups
        group_list = []

        assert out_channels % groups == 0

        for i in range(groups):
            offset = i * ig
            group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
            group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                    use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
        return Concatenate(name='%s/concat' % name)(group_list)


    def channel_shuffle(x, groups):
        """

        Parameters
        ----------
        x:
            Input tensor of with `channels_last` data format
        groups: int
            number of groups per channel


        Returns
        -------
            channel shuffled output tensor


        Examples
        --------
        Example for a 1D Array with 3 groups

        >>> d = np.array([0,1,2,3,4,5,6,7,8])
        >>> x = np.reshape(d, (3,3))
        >>> x = np.transpose(x, [1,0])
        >>> x = np.reshape(x, (9,))
        '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'


        """
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups

        x = K.reshape(x, [-1, height, width, groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])

        return x

    def _conv_blockSSD(inputs, filters,block_id=11):
    	channel_axis = -1
    	x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d_1' % block_id)(inputs)
    	x = Conv2D(filters, (1,1),padding='valid',use_bias=False,strides=(1, 1),name='conv__%d_1'%block_id)(x)
    	x = BatchNormalization(axis=channel_axis, name='conv_%d_bn_1'% block_id)(x)
    	x = Activation('relu', name='conv_%d_relu_1'% block_id)(x)
    	Conv = Conv2D(filters*2, (3,3), padding='valid', use_bias=False, strides=(2, 2), name='conv__%d_2' % block_id)(x)
    	x = BatchNormalization(axis=channel_axis, name='conv_%d_bn_2' % block_id)(Conv)
    	x = Activation('relu', name='conv_%d_relu_2' % block_id)(x)
    	return x,Conv
    
    ############################################################################
    # Build the network.
    ############################################################################

    if input_tensor != None:
        x = Input(tensor=input_tensor, shape=(img_height, img_width, img_channels))
    else:
        x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    num_shuffle_units = list([3,7,3])
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage *= scale_factor
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage = out_channels_in_stage.astype(int)

    # Get shufflenet architecture
    shufflenetv1 = shufflenet.ShuffleNet(groups=groups, 
                                  scale_factor=scale_factor,
                                  input_shape=(img_height, img_width, img_channels), 
                                  include_top=False)
    FeatureExtractor = Model(inputs=shufflenetv1.input, outputs=shufflenetv1.get_layer('stage3/block8/add').output)

    # Stage 3 last block unit
    shuffle_unit13 = FeatureExtractor(x1)
    layer = Activation('relu', name='stage3/block8/relu_out')(shuffle_unit13)

    layer = _shuffle_unit(layer, in_channels=out_channels_in_stage[4-2],
                      out_channels=out_channels_in_stage[4-1], strides=2,
                      groups=groups, bottleneck_ratio=0.25,
                      stage=4, block=1)
    layer = _shuffle_unit(layer, in_channels=out_channels_in_stage[4 - 1],
                          out_channels=out_channels_in_stage[4 - 1], strides=1,
                          groups=groups, bottleneck_ratio=0.25,
                          stage=4, block=2)
    layer = _shuffle_unit(layer, in_channels=out_channels_in_stage[4 - 1],
                          out_channels=out_channels_in_stage[4 - 1], strides=1,
                          groups=groups, bottleneck_ratio=0.25,
                          stage=4, block=3)
    layer, shuffle_unit17 = _shuffle_unit(layer, in_channels=out_channels_in_stage[4 - 1],
                          out_channels=out_channels_in_stage[4 - 1], strides=1,
                          groups=groups, bottleneck_ratio=0.25,
                          stage=4, block=4, linear=True)
    
    layer, conv18_2 = _conv_blockSSD(layer, 256, block_id=18)
    layer, conv19_2 = _conv_blockSSD(layer, 128, block_id=19)
    layer, conv20_2 = _conv_blockSSD(layer, 128, block_id=20)
    layer, conv21_2 = _conv_blockSSD(layer, 64, block_id=21)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv13_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', name='conv13_mbox_conf')(shuffle_unit13)
    conv17_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', name='conv17_mbox_conf')(shuffle_unit17)
    conv18_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', name='conv18_2_mbox_conf')(conv18_2)
    conv19_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', name='conv19_2_mbox_conf')(conv19_2)
    conv20_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', name='conv20_2_mbox_conf')(conv20_2)
    conv21_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', name='conv21_2_mbox_conf')(conv21_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv13_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv13_mbox_loc')(shuffle_unit13)
    conv17_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='conv17_mbox_loc')(shuffle_unit17)
    conv18_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv18_2_mbox_loc')(conv18_2)
    conv19_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', name='conv19_2_mbox_loc')(conv19_2)
    conv20_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv20_2_mbox_loc')(conv20_2)
    conv21_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv21_2_mbox_loc')(conv21_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv13_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                      two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                      variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv13_mbox_priorbox')(conv13_mbox_loc)
    conv17_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                      two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                      variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv17_mbox_priorbox')(conv17_mbox_loc)
    conv18_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv18_2_mbox_priorbox')(conv18_2_mbox_loc)
    conv19_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv19_2_mbox_priorbox')(conv19_2_mbox_loc)
    conv20_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv20_2_mbox_priorbox')(conv20_2_mbox_loc)
    conv21_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv21_2_mbox_priorbox')(conv21_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv13_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(conv13_mbox_conf)
    conv17_mbox_conf_reshape = Reshape((-1, n_classes), name='conv17_mbox_conf_reshape')(conv17_mbox_conf)
    conv18_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv18_2_mbox_conf_reshape')(conv18_2_mbox_conf)
    conv19_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv19_2_mbox_conf_reshape')(conv19_2_mbox_conf)
    conv20_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv20_2_mbox_conf_reshape')(conv20_2_mbox_conf)
    conv21_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv21_2_mbox_conf_reshape')(conv21_2_mbox_conf)
    
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv13_mbox_loc_reshape = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(conv13_mbox_loc)
    conv17_mbox_loc_reshape = Reshape((-1, 4), name='conv17_mbox_loc_reshape')(conv17_mbox_loc)
    conv18_2_mbox_loc_reshape = Reshape((-1, 4), name='conv18_2_mbox_loc_reshape')(conv18_2_mbox_loc)
    conv19_2_mbox_loc_reshape = Reshape((-1, 4), name='conv19_2_mbox_loc_reshape')(conv19_2_mbox_loc)
    conv20_2_mbox_loc_reshape = Reshape((-1, 4), name='conv20_2_mbox_loc_reshape')(conv20_2_mbox_loc)
    conv21_2_mbox_loc_reshape = Reshape((-1, 4), name='conv21_2_mbox_loc_reshape')(conv21_2_mbox_loc)
    
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv13_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_mbox_priorbox_reshape')(conv13_mbox_priorbox)
    conv17_mbox_priorbox_reshape = Reshape((-1, 8), name='conv17_mbox_priorbox_reshape')(conv17_mbox_priorbox)
    conv18_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv18_2_mbox_priorbox_reshape')(conv18_2_mbox_priorbox)
    conv19_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv19_2_mbox_priorbox_reshape')(conv19_2_mbox_priorbox)
    conv20_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv20_2_mbox_priorbox_reshape')(conv20_2_mbox_priorbox)
    conv21_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv21_2_mbox_priorbox_reshape')(conv21_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv13_mbox_conf_reshape,
                                                       conv17_mbox_conf_reshape,
                                                       conv18_2_mbox_conf_reshape,
                                                       conv19_2_mbox_conf_reshape,
                                                       conv20_2_mbox_conf_reshape,
                                                       conv21_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv13_mbox_loc_reshape,
                                                     conv17_mbox_loc_reshape,
                                                     conv18_2_mbox_loc_reshape,
                                                     conv19_2_mbox_loc_reshape,
                                                     conv20_2_mbox_loc_reshape,
                                                     conv21_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv13_mbox_priorbox_reshape,
                                                               conv17_mbox_priorbox_reshape,
                                                               conv18_2_mbox_priorbox_reshape,
                                                               conv19_2_mbox_priorbox_reshape,
                                                               conv20_2_mbox_priorbox_reshape,
                                                               conv21_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               #normalize_coords=normalize_coords, #change this parameter for inference
                                               normalize_coords=False,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    return model
