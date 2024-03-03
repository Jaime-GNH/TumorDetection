import os
import numpy as np
os.environ['KERAS_BACKEND'] = 'torch'
from patchify import patchify, unpatchify
import torch.nn.functional as tfn
from torch.utils.data import DataLoader

import keras
from keras import layers as kl
from keras import activations as ka
from keras import callbacks as kc
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy, MeanIoU


class EFSNet_keras(keras.Model, BaseClass):
    """
    Efficient Segmentation Network from https://ieeexplore.ieee.org/document/9063469
    """
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = self._default_config(EFSNetInit, **kwargs)
        self.verbose = kwargs.get('verbose')
        self.name = kwargs.get('name')
        self.input_shape = kwargs.get('input_shape')
        self.num_classes = kwargs.get('num_classes')
        self.pos_weight = kwargs.get('pos_weight')
        self.loss_seg = self.custom_bce
        self.loss_lab = torch.nn.CrossEntropyLoss(
            ignore_index=kwargs.get('ignore_index'),
            weight=torch.as_tensor(kwargs.get('class_weights'),
                                   device=Device.get('device')))
        self.seg_loss_tracker = keras.metrics.Mean(name="seg_loss")
        self.lab_loss_tracker = keras.metrics.Mean(name="lab_loss")

        self.efs_kwargs = kwargs.get('EfficientNetInit')
        self.built = True
        self.loaded = False

    @property
    def metrics(self):
        return [self.seg_loss_tracker, self.lab_loss_tracker]

    def load_model(self):
        """
        """
        assert self.compiled, f'Model must be compiled first.'
        self.load_weights(os.path.join(ReportingPathDir.get('dir_path'),
                                       'models',
                                       self.name + '.weights.h5'))
        if self.verbose > 0:
            print('Loaded model successfully')
        self.loaded = True

    def call(self, inputs):
        inp = kl.Input(self.input_shape)
        x1, x2, x3 = self.encoder(inp)
        x = self.decoder(x1, x2, x3)
        x3 = self._downsampling_block(x, filters=8,
                                      activation_layer=self.efs_kwargs.get('activation_layer'),
                                      dr_rate=self.efs_kwargs.get('dr_rate'),
                                      use_bias=self.efs_kwargs.get('use_bias'),
                                      kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                      name='label_downsample')
        x3 = kl.Flatten(name='label_flatten')(x3)
        label = kl.Dense(self.num_classes,
                         kernel_initializer=self.efs_kwargs.get('kernel_initializer'), name='label')(x3)
        segment = self.conv2dtranspose(x, 2, kernel_size=(3, 3), strides=2, padding=1,
                                       name='mask')

        return segment, label

    def encoder(self, x):
        outputs = []
        x = self._initial_block(x,
                                activation_layer=self.efs_kwargs.get('activation_layer'),
                                dr_rate=self.efs_kwargs.get('dr_rate'),
                                use_bias=self.efs_kwargs.get('use_bias'),
                                kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                name='E_IB')
        x = self._downsampling_block(x, filters=self.efs_kwargs.get('max_filters') // 2,
                                     activation_layer=self.efs_kwargs.get('activation_layer'),
                                     dr_rate=self.efs_kwargs.get('dr_rate'),
                                     use_bias=self.efs_kwargs.get('use_bias'),
                                     kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                     name='E_DS1')
        outputs.append(x)
        for fb in range(self.efs_kwargs.get('num_factorized_blocks')):
            x = self._factorized_block(x, filters=self.efs_kwargs.get('max_filters') // 2,
                                       activation_layer=self.efs_kwargs.get('activation_layer'),
                                       dr_rate=self.efs_kwargs.get('dr_rate'),
                                       use_bias=self.efs_kwargs.get('use_bias'),
                                       kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                       name=f'E_FB{fb + 1}')
        x = self._downsampling_block(x, filters=self.efs_kwargs.get('max_filters'),
                                     activation_layer=self.efs_kwargs.get('activation_layer'),
                                     dr_rate=self.efs_kwargs.get('dr_rate'),
                                     use_bias=self.efs_kwargs.get('use_bias'),
                                     kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                     name='E_DS2')
        outputs.append(x)
        for sdc in range(self.efs_kwargs.get('num_super_sdc_blocks')):
            for k in range(4):
                x = self._sdc_block(x, filters=self.efs_kwargs.get('max_filters'),
                                    groups=self.efs_kwargs.get('groups'),
                                    dilation=2 ** k,
                                    activation_layer=self.efs_kwargs.get('activation_layer'),
                                    dr_rate=self.efs_kwargs.get('dr_rate'),
                                    use_bias=self.efs_kwargs.get('use_bias'),
                                    kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                    name=f'E_SDC{sdc + 1}{k + 1}')
        outputs.append(x)
        return outputs

    def decoder(self, x1, x2, x3):
        x = self._upsampling_module(x3, x2, self.efs_kwargs.get('max_filters') // 4,
                                    activation_layer=self.efs_kwargs.get('activation_layer'),
                                    dr_rate=self.efs_kwargs.get('dr_rate'),
                                    use_bias=self.efs_kwargs.get('use_bias'),
                                    kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                    name='D_UM1')
        for sh in range(2):
            x = self._shufflenet(x, filters=self.efs_kwargs.get('max_filters') // 2, groups=self.efs_kwargs.get('groups'),
                                 activation_layer=self.efs_kwargs.get('activation_layer'),
                                 dr_rate=self.efs_kwargs.get('dr_rate'),
                                 use_bias=self.efs_kwargs.get('use_bias'),
                                 kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                 name=f'D_SN1{sh + 1}')

        x = self._upsampling_module(x, x1, self.efs_kwargs.get('max_filters') // 16,
                                    activation_layer=self.efs_kwargs.get('activation_layer'),
                                    dr_rate=self.efs_kwargs.get('dr_rate'),
                                    use_bias=self.efs_kwargs.get('use_bias'),
                                    kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                    name='D_UM2')
        for sh in range(2):
            x = self._shufflenet(x, filters=16, groups=self.efs_kwargs.get('groups') // 2,
                                 dr_rate=self.efs_kwargs.get('dr_rate'),
                                 activation_layer=self.efs_kwargs.get('activation_layer'),
                                 use_bias=self.efs_kwargs.get('use_bias'),
                                 kernel_initializer=self.efs_kwargs.get('kernel_initializer'),
                                 name=f'D_SN2{sh + 1}')
        return x

    @staticmethod
    def conv2dtranspose(x, filters, kernel_size, strides, padding, name):

        x = kl.Permute((3, 1, 2), name=name+'_Perm1')(x)
        x = kl.TorchModuleWrapper(torch.nn.ConvTranspose2d(x.shape[1], out_channels=filters, kernel_size=kernel_size,
                                                           stride=strides, padding=padding, output_padding=padding),
                                  name=name + '_Conv2dT_1')(x)
        x = kl.Permute((2, 3, 1), name=name+'_Perm2')(x)
        return x

    @staticmethod
    def _initial_block(x, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=16 - x.shape[-1], kernel_size=(3, 3), strides=2, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)
        x2 = kl.MaxPool2D(pool_size=(2, 2), name=name + '_MaxPool_21')(x)
        # x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_21')(x2)
        # x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        # x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)
        return kl.Concatenate(axis=-1, name=name + '_Concat12')([x1, x2])

    @staticmethod
    def _downsampling_block(x, filters, activation_layer, dr_rate, use_bias,
                            kernel_initializer, name):
        x1 = kl.MaxPool2D(pool_size=(2, 2), name=name + '_MaxPool_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        # x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        x2 = kl.Conv2D(filters=filters // 4, kernel_size=(2, 2), strides=2, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_21')(x)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_21')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        x2 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_22')(x2)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_22')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_22')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_22')(x2))
        x2 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_23')(x2)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_23')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_23')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_23')(x2))
        x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x1, x2]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x1, x2])))

    @staticmethod
    def _factorized_block(x, filters, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 3), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_13')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_12')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_14')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_14')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_13')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_13')(x1))
        x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_1')(x1)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x, x1]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x, x1])))

    def _sdc_block(self, x, filters, groups, dilation, activation_layer, dr_rate, use_bias,
                   kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        x1 = kl.Lambda(self.channel_shuffle, arguments={'groups': groups}, trainable=False,
                       name=name + '_ChannelShuffle_11')(x1)
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 3), dilation_rate=dilation,
                       padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_13')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_12')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_1')(x1)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x, x1]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x, x1])))

    @staticmethod
    def _upsampling_module(x1, x2, filters, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x1)
        x1 = kl.BatchNormalization(axis=-1, name=name + '_BatchNorm_11')(x1)
        x1 = kl.UpSampling2D(size=(2, 2), interpolation='bilinear', name=name + '_Upsample2D_11')(x1)
        # x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        # x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        x2 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Upsample2D_21')(x2)
        x2 = kl.BatchNormalization(axis=-1, name=name + '_BatchNorm_21')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        x2 = kl.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=2, padding='same', use_bias=use_bias,
                                kernel_initializer=kernel_initializer, name=name + '_Conv2DTrans_21')(x2)
        # x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_22')(x2)
        # x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_22')(x2)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_22')(x2))
        # x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)

        return (kl.LeakyReLU(0.1,
                             name=name + '_LeakyReLU_out')(kl.Concatenate(name=name + '_Concat_out')([x1, x2]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3],
                         name=name + '_PReLU_out')(kl.Concatenate(name=name + '_Concat_out')([x1, x2])))

    def _shufflenet(self, x, filters, groups, activation_layer, dr_rate, use_bias, kernel_initializer, name):

        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = ka.relu(x1)
        x1 = kl.Lambda(self.channel_shuffle, arguments={'groups': groups}, trainable=False,
                       name=name + '_ChannelShuffle_11')(x1)
        x1 = kl.SeparableConv2D(filters=filters // 4, kernel_size=(3, 3),
                                padding='same', use_bias=use_bias,
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=name + '_SeparableConv2d_11')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        return ka.relu(kl.Add(name=name + '_Add_out')([x, x1]))

    @staticmethod
    def channel_shuffle(x, groups):
        """

        :param x:
        :param groups:
        :return:
        """
        batch, height, width, in_channels = x.shape
        channels_per_group = in_channels // groups

        x = kl.Reshape([height, width, groups, channels_per_group])(x)
        x = kl.Permute((1, 2, 4, 3))(x)  # transpose
        x = kl.Reshape([height, width, in_channels])(x)

        return x

    def custom_bce(self, y_true, y_pred):
        loss_mask = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor([self.pos_weight] * y_pred.shape[-1])[None, None, :])
        lm = loss_mask(y_pred,
                       tfn.one_hot(torch.from_numpy(y_true).to(torch.long).squeeze(),
                                   num_classes=2).double())
        return lm

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, mask, label = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        mask_pred, label_pred = self(x, training=True)  # Forward pass
        loss_seg = self.loss_seg(y_true=mask, y_pred=mask_pred)
        loss_lab = self.loss_lab(label_pred, label)
        loss = loss_seg + loss_lab
        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': loss},
                **{m.name: m.result() for m in self.metrics}}


class EFSNetModel(BaseClass):
    """
    Wrapper for usage of EFSNet from https://ieeexplore.ieee.org/document/9063469
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__()
        # kwargs = self._default_config(EFSNetModelInit, **kwargs)
        self.verbose = kwargs.get('verbose')
        self.input_shape = kwargs.get('input_shape')
        self.num_classes = kwargs.get('num_classes')
        self.compile_params = kwargs.get('compile_params')
        self.train_params = kwargs.get('train_params')
        self.model_name = kwargs.get('model_name')
        self.loaded = False
        if kwargs.get('precision') is not None:
            keras.mixed_precision.set_dtype_policy(kwargs.get('precision'))
        self.model = self.build_model(**kwargs.get('EfficientNetInit'))

    def build_model(self, **kwargs):
        """
        Builds model
        """
        inp = kl.Input(self.input_shape)
        x1, x2, x3 = self.encoder(inp, **kwargs)
        x = self.decoder(x1, x2, x3, **kwargs)
        x3 = kl.Flatten(name='flatten')(x3)
        label = kl.Dense(self.num_classes, kernel_initializer=kwargs.get('kernel_initializer'), name='label')(x3)
        segment = self.conv2dtranspose(x, 2, kernel_size=(3, 3), strides=2, padding=1,
                                       name='mask')
        model = keras.Model([inp], [segment, label], name=self.model_name)
        return model

    def encoder(self, x, **kwargs):
        outputs = []
        x = self._initial_block(x,
                                activation_layer=kwargs.get('activation_layer'),
                                dr_rate=kwargs.get('dr_rate'),
                                use_bias=kwargs.get('use_bias'),
                                kernel_initializer=kwargs.get('kernel_initializer'),
                                name='E_IB')
        x = self._downsampling_block(x, filters=kwargs.get('max_filters') // 2,
                                     activation_layer=kwargs.get('activation_layer'),
                                     dr_rate=kwargs.get('dr_rate'),
                                     use_bias=kwargs.get('use_bias'),
                                     kernel_initializer=kwargs.get('kernel_initializer'),
                                     name='E_DS1')
        outputs.append(x)
        for fb in range(kwargs.get('num_factorized_blocks')):
            x = self._factorized_block(x, filters=kwargs.get('max_filters') // 2,
                                       activation_layer=kwargs.get('activation_layer'),
                                       dr_rate=kwargs.get('dr_rate'),
                                       use_bias=kwargs.get('use_bias'),
                                       kernel_initializer=kwargs.get('kernel_initializer'),
                                       name=f'E_FB{fb + 1}')
        x = self._downsampling_block(x, filters=kwargs.get('max_filters'),
                                     activation_layer=kwargs.get('activation_layer'),
                                     dr_rate=kwargs.get('dr_rate'),
                                     use_bias=kwargs.get('use_bias'),
                                     kernel_initializer=kwargs.get('kernel_initializer'),
                                     name='E_DS2')
        outputs.append(x)
        for sdc in range(kwargs.get('num_super_sdc_blocks')):
            for k in range(4):
                x = self._sdc_block(x, filters=kwargs.get('max_filters'),
                                    groups=kwargs.get('groups'),
                                    dilation=2 ** k,
                                    activation_layer=kwargs.get('activation_layer'),
                                    dr_rate=kwargs.get('dr_rate'),
                                    use_bias=kwargs.get('use_bias'),
                                    kernel_initializer=kwargs.get('kernel_initializer'),
                                    name=f'E_SDC{sdc + 1}{k + 1}')
        outputs.append(x)
        return outputs

    def decoder(self, x1, x2, x3, **kwargs):
        x = self._upsampling_module(x3, x2, kwargs.get('max_filters') // 4,
                                    activation_layer=kwargs.get('activation_layer'),
                                    dr_rate=kwargs.get('dr_rate'),
                                    use_bias=kwargs.get('use_bias'),
                                    kernel_initializer=kwargs.get('kernel_initializer'),
                                    name='D_UM1')
        for sh in range(2):
            x = self._shufflenet(x, filters=kwargs.get('max_filters') // 2, groups=kwargs.get('groups'),
                                 activation_layer=kwargs.get('activation_layer'),
                                 dr_rate=kwargs.get('dr_rate'),
                                 use_bias=kwargs.get('use_bias'),
                                 kernel_initializer=kwargs.get('kernel_initializer'),
                                 name=f'D_SN1{sh + 1}')

        x = self._upsampling_module(x, x1, kwargs.get('max_filters') // 16,
                                    activation_layer=kwargs.get('activation_layer'),
                                    dr_rate=kwargs.get('dr_rate'),
                                    use_bias=kwargs.get('use_bias'),
                                    kernel_initializer=kwargs.get('kernel_initializer'),
                                    name='D_UM2')
        for sh in range(2):
            x = self._shufflenet(x, filters=16, groups=kwargs.get('groups') // 2,
                                 dr_rate=kwargs.get('dr_rate'),
                                 activation_layer=kwargs.get('activation_layer'),
                                 use_bias=kwargs.get('use_bias'),
                                 kernel_initializer=kwargs.get('kernel_initializer'),
                                 name=f'D_SN2{sh + 1}')
        return x

    @staticmethod
    def conv2dtranspose(x, filters, kernel_size, strides, padding, name):

        x = kl.Permute((3, 1, 2), name=name+'_Perm1')(x)
        x = kl.TorchModuleWrapper(torch.nn.ConvTranspose2d(x.shape[1], out_channels=filters, kernel_size=kernel_size,
                                                           stride=strides, padding=padding, output_padding=padding),
                                  name=name + '_Conv2dT_1')(x)
        x = kl.Permute((2, 3, 1), name=name+'_Perm2')(x)
        return x

    @staticmethod
    def _initial_block(x, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=16 - x.shape[-1], kernel_size=(3, 3), strides=2, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)
        x2 = kl.MaxPool2D(pool_size=(2, 2), name=name + '_MaxPool_21')(x)
        # x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_21')(x2)
        # x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        # x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)
        return kl.Concatenate(axis=-1, name=name + '_Concat12')([x1, x2])

    @staticmethod
    def _downsampling_block(x, filters, activation_layer, dr_rate, use_bias,
                            kernel_initializer, name):
        x1 = kl.MaxPool2D(pool_size=(2, 2), name=name + '_MaxPool_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        # x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        x2 = kl.Conv2D(filters=filters // 4, kernel_size=(2, 2), strides=2, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_21')(x)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_21')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        x2 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_22')(x2)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_22')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_22')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_22')(x2))
        x2 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_23')(x2)
        x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_23')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_23')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_23')(x2))
        x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x1, x2]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x1, x2])))

    @staticmethod
    def _factorized_block(x, filters, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 3), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_13')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_12')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_14')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_14')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_13')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_13')(x1))
        x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_1')(x1)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x, x1]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x, x1])))

    def _sdc_block(self, x, filters, groups, dilation, activation_layer, dr_rate, use_bias,
                   kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        x1 = kl.Lambda(self.channel_shuffle, arguments={'groups': groups}, trainable=False,
                       name=name + '_ChannelShuffle_11')(x1)
        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(3, 3), dilation_rate=dilation,
                       padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_13')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_12')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_1')(x1)

        return (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_out')(kl.Add(name=name + '_Add_out')([x, x1]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_out')(kl.Add(name=name + '_Add_out')([x, x1])))

    @staticmethod
    def _upsampling_module(x1, x2, filters, activation_layer, dr_rate, use_bias, kernel_initializer, name):
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x1)
        x1 = kl.BatchNormalization(axis=-1, name=name + '_BatchNorm_11')(x1)
        x1 = kl.UpSampling2D(size=(2, 2), interpolation='bilinear', name=name + '_Upsample2D_11')(x1)
        # x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        # x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_12')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        x2 = kl.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Upsample2D_21')(x2)
        x2 = kl.BatchNormalization(axis=-1, name=name + '_BatchNorm_21')(x2)
        x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_21')(x2)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_21')(x2))
        x2 = kl.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=2, padding='same', use_bias=use_bias,
                                kernel_initializer=kernel_initializer, name=name + '_Conv2DTrans_21')(x2)
        # x2 = kl.BatchNormalization(-1, name=name + '_BatchNorm_22')(x2)
        # x2 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_22')(x2)
        #       if activation_layer == 'leakyrelu' else
        #       kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_22')(x2))
        # x2 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_21')(x2)

        return (kl.LeakyReLU(0.1,
                             name=name + '_LeakyReLU_out')(kl.Concatenate(name=name + '_Concat_out')([x1, x2]))
                if activation_layer == 'leakyrelu' else
                kl.PReLU(shared_axes=[1, 2, 3],
                         name=name + '_PReLU_out')(kl.Concatenate(name=name + '_Concat_out')([x1, x2])))

    def _shufflenet(self, x, filters, groups, activation_layer, dr_rate, use_bias, kernel_initializer, name):

        x1 = kl.Conv2D(filters=filters // 4, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_11')(x)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_11')(x1)
        x1 = ka.relu(x1)
        x1 = kl.Lambda(self.channel_shuffle, arguments={'groups': groups}, trainable=False,
                       name=name + '_ChannelShuffle_11')(x1)
        x1 = kl.SeparableConv2D(filters=filters // 4, kernel_size=(3, 3),
                                padding='same', use_bias=use_bias,
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=name + '_SeparableConv2d_11')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_12')(x1)
        x1 = kl.Conv2D(filters=filters, kernel_size=(1, 1), groups=groups, padding='same', use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=name + '_Conv2D_12')(x1)
        x1 = kl.BatchNormalization(-1, name=name + '_BatchNorm_13')(x1)
        x1 = (kl.LeakyReLU(0.1, name=name + '_LeakyReLU_11')(x1)
              if activation_layer == 'leakyrelu' else
              kl.PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_11')(x1))
        # x1 = kl.SpatialDropout2D(rate=dr_rate, name=name + '_SpatialDropout_11')(x1)

        return ka.relu(kl.Add(name=name + '_Add_out')([x, x1]))

    @staticmethod
    def channel_shuffle(x, groups):
        """

        :param x:
        :param groups:
        :return:
        """
        batch, height, width, in_channels = x.shape
        channels_per_group = in_channels // groups

        x = kl.Reshape([height, width, groups, channels_per_group])(x)
        x = kl.Permute((1, 2, 4, 3))(x)  # transpose
        x = kl.Reshape([height, width, in_channels])(x)

        return x

    @staticmethod
    def calculate_sample_weight(y):
        """

        :param y:
        :return:
        """
        sample_weight = np.sum(np.array([1 / np.log(1.02 + np.sum(y[..., i],
                                                                  axis=(1, 2)) / (y.shape[1] * y.shape[0]))
                                         for i in range(y.shape[-1])]), axis=0)
        return sample_weight - np.min(sample_weight)

    def custom_bce(self, y_true, y_pred):
        loss_mask = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor([self.compile_params.get('pos_weight')] * y_pred.shape[-1],
                                       device=y_pred.device)[None, None, :])
        lm = loss_mask(y_pred,
                       tfn.one_hot(y_true.to(torch.long).squeeze(),
                                   num_classes=2).double())
        return lm

    # def weighted_categorical_crossentropy(self, y_true, y_pred):
    #     assert len(y_pred.shape) == 4, f'Weighted Categorical Crossentropy works with (N, H, W, C) predicted images'
    #     assert y_pred.shape[-1] == self.num_classes, f'Classes must be in last position.'
    #     assert y_true.shape == y_pred.shape[:3], ('y_true shape must be equal to y_pred shape without classes. '
    #                                               f'Got {y_true.shape}, {y_pred.shape}')
    #     loss = torch.nn.CrossEntropyLoss(ignore_index=self.compile_params.get('ignore_index'),
    #                                      weight=torch.as_tensor(self.compile_params.get('class_weights'),
    #                                                             device=y_pred.device),
    #                                      reduction='sum')
    #     return loss(y_pred.permute(0, 3, 1, 2), y_true.to(torch.long))/y_pred.numel()

    def compile_model(self):
        """
        Compiles the model
        """
        self.model.compile(
            optimizer=self.compile_params.get('optimizer'),
            loss=[
                self.custom_bce,
                # None,
                SparseCategoricalCrossentropy(from_logits=True, ignore_class=self.compile_params.get('ignore_index')),
                # torch.nn.CrossEntropyLoss(ignore_index=self.compile_params.get('ignore_index'),
                #                           weight=torch.as_tensor(self.compile_params.get('class_weights'),
                #                                                  device=Device.get('device')))
            ],
            # metrics=[SparseCategoricalAccuracy(),
                     # MeanIoU(num_classes=self.num_classes,
                     #         sparse_y_pred=False)
                     # ]
        )
        if self.verbose > 2:
            self.model.summary(expand_nested=True)

    def train_model(self, resume_training=False, **kwargs):
        """
        Trains model
        :param resume_training: False
        :return: history, fig train metrics
        """
        if not self.model.compiled:
            self.compile_model()
        if resume_training:
            self.load_model()

        callbacks = []
        if self.train_params.get('use_earlystopping'):
            callbacks.append(kc.EarlyStopping(**self.train_params.get('es_kwargs')))
        if self.train_params.get('use_reducelronplateau'):
            callbacks.append(kc.ReduceLROnPlateau(**self.train_params.get('rlr_kwargs')))
        if self.train_params.get('use_modelcheckpoint'):
            os.makedirs(os.path.join(ReportingPathDir.get('dir_path'), 'models'), exist_ok=True)
            callbacks.append(kc.ModelCheckpoint(
                filepath=os.path.join(ReportingPathDir.get('dir_path'), 'models', self.model.name + '.weights.h5'),
                **self.train_params.get('ckpt_params')))
        if self.train_params.get('log_tensorboard'):
            os.makedirs(os.path.join(ReportingPathDir.get('dir_path'), 'models'), exist_ok=True)
            callbacks.append(kc.TensorBoard(
                log_dir=os.path.join(ReportingPathDir.get('dir_path'), 'models'),
                update_freq='batch'
            ))
        if all([d in kwargs for d in ['x_train', 'x_test', 'y_train', 'y_test']]):
            history = self.model.fit(
                kwargs.get('x_train'),
                kwargs.get('y_train'),
                epochs=self.train_params.get('epochs'),
                batch_size=self.train_params.get('batch_size'),
                steps_per_epoch=self.train_params.get('steps_per_epoch'),
                shuffle=self.train_params.get('shuffle'),
                validation_steps=self.train_params.get('validation_steps'),
                validation_data=(kwargs.get('x_val'), kwargs.get('y_val')),
                callbacks=callbacks
            )
        elif all([d in kwargs for d in ['dataset_train', 'dataset_test']]):
            train_dataloader = DataLoader(kwargs.get('dataset_train'),
                                          self.train_params.get('batch_size'),
                                          shuffle=self.train_params.get('shuffle'),
                                          )
            val_dataloader = DataLoader(kwargs.get('dataset_test'),
                                        self.train_params.get('batch_size'),
                                        shuffle=self.train_params.get('shuffle'),
                                        )
            history = self.model.fit(
                train_dataloader,
                epochs=self.train_params.get('epochs'),
                steps_per_epoch=self.train_params.get('steps_per_epoch'),
                validation_steps=self.train_params.get('validation_steps'),
                validation_data=val_dataloader,
                callbacks=callbacks
            )
        else:
            raise ValueError('''
            You must past either a [x_train, x_test, y_train, y_test] or a [dataset_train, dataset_test] to
            the keyword arguments.
            ''')

        return history

    def load_model(self):
        """
        """
        if not self.model.compiled:
            self.compile_model()
        self.model.load_weights(os.path.join(ReportingPathDir.get('dir_path'),
                                             'models',
                                             self.model.name + '.weights.h5'))
        if self.verbose > 0:
            print('Loaded model successfully')
        self.loaded = True

    def composed_inference(self, image, patch_step):
        """

        :return: prediction
        """
        if not self.loaded:
            self.load_model()

        patch_dim = self.model.input.shape[1:-1]
        batch = patchify(image,
                         patch_dim,
                         patch_step).reshape(-1, *patch_dim, 1)
        y = self.model.predict(batch)
        return unpatchify(y, (*image.shape, self.num_classes))

    def predict_batch(self, batch):
        """

        :param batch:
        :return:
        """
        if not self.loaded:
            self.load_model()
        assert len(batch.shape) == 4, f'image shape must have 4 dimensions (B,H,W,C). Got {batch.shape}'
        assert batch.shape[1:] == self.input_shape, (f'Image dimensions must be batch times {self.input_shape}.'
                                                     f' Got {batch.shape[1:]}')
        mask = np.argmax(self.model(batch), axis=-1).squeeze()
        return mask


def compile_model(model, optimizer):
    """
    Compiles the model
    """
    model.compile(
        optimizer=optimizer
    )
    return model


def train_model(model, dataset_train, dataset_test, **kwargs):
    """
    Trains model
    :param model: keras.Model
    :param resume_training: False
    :return: history, fig train metrics
    """
    if not model.compiled:
        model = compile_model(model, kwargs.get('optimizer'))
    if kwargs.get('resume_training', False):
        model.load_model()

    callbacks = []
    if kwargs.get('use_modelcheckpoint', True):
        os.makedirs(os.path.join(ReportingPathDir.get('dir_path'), 'models'), exist_ok=True)
        callbacks.append(kc.ModelCheckpoint(
            filepath=os.path.join(ReportingPathDir.get('dir_path'), 'models', model.name + '.weights.h5'),
            **kwargs.get('ckpt_params', dict(
                save_best_only=True,
                save_weights_only=True,
                verbose=1 if kwargs.get('verbose', 1) > 0 else 0)))
        )
    if kwargs.get('log_tensorboard', False):
        os.makedirs(os.path.join(ReportingPathDir.get('dir_path'), 'models'), exist_ok=True)
        callbacks.append(kc.TensorBoard(
            log_dir=os.path.join(ReportingPathDir.get('dir_path'), 'models'),
            update_freq='batch'
        ))
    train_dataloader = DataLoader(dataset_train,
                                  batch_size=kwargs.get('batch_size', 32),
                                  shuffle=kwargs.get('shuffle', True),
                                  )
    val_dataloader = DataLoader(dataset_test,
                                batch_size=kwargs.get('batch_size', 32),
                                shuffle=kwargs.get('shuffle', True),
                                )
    history = model.fit(
        train_dataloader,
        epochs=kwargs.get('epochs', 250),
        steps_per_epoch=kwargs.get('steps_per_epoch'),
        validation_steps=kwargs.get('validation_steps'),
        validation_data=val_dataloader,
        callbacks=callbacks
    )
    return history
