'''

author: ichernu@softserveinc.com

Based on example: https://github.com/aws-samples/amazon-sagemaker-brain-segmentation.
'''
from __future__ import print_function

import argparse
import bisect
import json
import logging
import os
import random
import time
from collections import Counter
from itertools import chain, islice
from PIL import Image
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, nd
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet.gluon import data
logging.basicConfig(level=logging.INFO)

    
# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = gluon.SymbolBlock.imports(
        '%s/model-symbol.json' % model_dir,
        ['data'],
        '%s/model-0000.params' % model_dir,
    )
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    try:
        input_data = json.loads(data)
        nda = mx.nd.array(input_data)
        nda *= 1.0/nda.max()
        output = net(nda)
        im =np.array(Image.fromarray((output.asnumpy()[0][0]* 255).astype('uint8'), mode='L')) 
        response_body = json.dumps(im.tolist())
    except Exception as e:
        logging.error(str(e))
        return  json.dumps([1,2]), output_content_type
    return response_body, output_content_type

IM_SIZE = 1024


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #
    
class ImageWithMaskDataset(data.Dataset):
    def __init__(
            self, 
            root,
            num_classes=None, 
            split_type='train',
    ):
        if split_type == 'train':
            images_dir = root + '/train/'
            masks_dir = root + '/train_annotation/'
        else:
            images_dir = root + '/validation/'
            masks_dir = root + '/validation_annotation/'
            
        self.ids = sorted(os.listdir(images_dir))
        self.idsm = sorted(os.listdir(masks_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.idsm]
    
    def __getitem__(self, i):
        # read data
        image = mx.image.imread(self.images_fps[i], 0)
        mask = mx.image.imread(self.masks_fps[i], 0)
        
        try:
            image = mx.image.imresize(image, IM_SIZE,IM_SIZE)
            # avoid averaging original values during interpolation for mask
            mask = mx.image.imresize(mask, IM_SIZE,IM_SIZE, interp=3)
        except Exception as e:
            print(self.images_fps[i])
            return self.__getitem__(i+1)


        
        # normalize images
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        image *= 1.0/image.max()
        mask *= 1.0/mask.max()
        image =  mx.nd.transpose(image, (2, 0, 1))
        mask = mx.nd.transpose(mask, (2, 0, 1))
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
def DataLoaderGenerator(data_loader):
    """
    A generator wrapper for loading images (with masks) from a 'ImageWithMaskDataset' dataset.

    Parameters
    ----------
    data_loader : 'Dataset' instance
        Instance of Gluon 'Dataset' object from which image / mask pairs are yielded.
    """
    for data, label in data_loader:
        data_desc = mx.io.DataDesc(name='data', shape=data.shape, dtype=np.float32)
        label_desc = mx.io.DataDesc(name='label', shape=label.shape, dtype=np.float32)
        batch = mx.io.DataBatch(
            data=[data],
            label=[label],
            provide_data=[data_desc],
            provide_label=[label_desc])
        yield batch


class DataLoaderIter(mx.io.DataIter):
    """
    An iterator wrapper for loading images (with masks) from an 'ImageWithMaskDataset' dataset.
    Allows for MXNet Module API to train using Gluon data loaders.

    Parameters
    ----------
    root : str
        Root directory containg image / mask pairs stored as `xyz.jpg` and `xyz_mask.png`.
    num_classes : int
        Number of classes in data set.
    batch_size : int
        Size of batch.
    shuffle : Bool
        Whether or not to shuffle data.
    num_workers : int
        Number of sub-processes to spawn for loading data. Default 0 means none.
    """
    def __init__(self, root, num_classes, batch_size, shuffle=False, num_workers=0, split_type='train'):

        self.batch_size = batch_size
        self.dataset = ImageWithMaskDataset(root=root, num_classes=num_classes, split_type=split_type)
        if mx.__version__ == "0.11.0":
            self.dataloader = mx.gluon.data.DataLoader(
                self.dataset, batch_size=batch_size, shuffle=shuffle, last_batch='rollover')
        else:
            self.dataloader = mx.gluon.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                last_batch='rollover')
        self.dataloader_generator = DataLoaderGenerator(self.dataloader)

    def __iter__(self):
        return self

    def reset(self):
        self.dataloader_generator = DataLoaderGenerator(self.dataloader)

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [
            mx.io.DataDesc(name='data', shape=(self.batch_size,) + self.dataset[0][0].shape, dtype=np.float32)
        ]

    @property
    def provide_label(self):
        return [
            mx.io.DataDesc(name='label', shape=(self.batch_size,) + self.dataset[0][1].shape, dtype=np.float32)
        ]

    def next(self):
        return next(self.dataloader_generator)
    
def train(args):
    logging.info(mx.__version__)
    
    # Get hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    model_dir = os.environ['SM_MODEL_DIR']
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    current_host = args.current_host
    hosts = args.hosts
    
    beta1 = 0.9 
    beta2 = 0.99 
    num_workers = args.num_workers
    num_classes = 1
    # Set context for compute based on instance environment
    if num_gpus > 0:
        ctx = [mx.gpu(i) for i in range(num_gpus)]
    else:
        ctx = mx.cpu()
        
    # Locate compressed training/validation data
    root_data_dir = args.root_dir
    
    # Define custom iterators on extracted data locations.
    train_iter = DataLoaderIter(
        root_data_dir,
        num_classes,
        batch_size,
        True,
        num_workers,
        'train')
    validation_iter = DataLoaderIter(
        root_data_dir,
        num_classes,
        batch_size,
        False,
        num_workers,
        'validation')
    
    # Build network symbolic graph
    sym = build_unet(num_classes)
    logging.info("Sym loaded")
    
    # Load graph into Module
    net = mx.mod.Module(sym, context=ctx, data_names=('data',), label_names=('label',))
    
    # Initialize Custom Metric
    dice_metric = mx.metric.CustomMetric(feval=avg_dice_coef_metric, allow_extra_outputs=True)
    logging.info("Starting model fit")
    
    # Start training the model
    net.fit(
        train_data=train_iter,
        eval_data=validation_iter,
        eval_metric=dice_metric,
        initializer=mx.initializer.Xavier(magnitude=6),
        optimizer='adam',
        optimizer_params={
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2},
        num_epoch=epochs)
    
    # Save Parameters
    net.save_params('params')
    
    # Build inference-only graphs, set parameters from training models
    sym = build_unet(num_classes, inference=True)
    net = mx.mod.Module(
        sym, context=ctx, data_names=(
            'data',), label_names=None)
    
    # Re-binding model for a batch-size of one
    net.bind(data_shapes=[('data', (1,) + train_iter.provide_data[0][1][1:])])
    net.load_params('params')
    # save model
    from sagemaker_mxnet_container.training_utils import save
    save(os.environ['SM_MODEL_DIR'], net)

# Architecture adapted from https://arxiv.org/abs/1505.04597

def _conv_block(inp, num_filter, kernel, pad, block, conv_block):
    conv = mx.sym.Convolution(inp, num_filter=num_filter, kernel=kernel, pad=pad, name='conv%i_%i' % (block, conv_block))
    conv = mx.sym.BatchNorm(conv, fix_gamma=True, name='bn%i_%i' % (block, conv_block))
    conv = mx.sym.Activation(conv, act_type='relu', name='relu%i_%i' % (block, conv_block))
    return conv

def _down_block(inp, num_filter, kernel, pad, block, pool=True):
    conv = _conv_block(inp, num_filter, kernel, pad, block, 1)
    conv = _conv_block(conv, num_filter, kernel, pad, block, 2)
    if pool:
        pool = mx.sym.Pooling(conv, kernel=(2, 2), stride=(2, 2), pool_type='max', name='pool_%i' % block)
        return pool, conv
    return conv

def _down_branch(inp):
    pool1, conv1 = _down_block(inp, num_filter=32, kernel=(3, 3), pad=(1, 1), block=1)
    pool2, conv2 = _down_block(pool1, num_filter=64, kernel=(3, 3), pad=(1, 1), block=2)
    pool3, conv3 = _down_block(pool2, num_filter=128, kernel=(3, 3), pad=(1, 1), block=3)
    pool4, conv4 = _down_block(pool3, num_filter=256, kernel=(3, 3), pad=(1, 1), block=4)
    conv5 = _down_block(pool4, num_filter=512, kernel=(3, 3), pad=(1, 1), block=5, pool=False)
    return [conv5, conv4, conv3, conv2, conv1]

def _up_block(inp, down_feature, num_filter, kernel, pad, block):
    trans_conv = mx.sym.Deconvolution(
        inp, num_filter=num_filter, kernel=(2, 2), stride=(2, 2), no_bias=True, name='trans_conv_%i' % block
    )
    up = mx.sym.concat(*[trans_conv, down_feature], dim=1, name='concat_%i' % block)
    conv = _conv_block(up, num_filter, kernel, pad, block, 1)
    conv = _conv_block(conv, num_filter, kernel, pad, block, 2)
    return conv

def _up_branch(down_features, num_classes):
    conv6 = _up_block(down_features[0], down_features[1], num_filter=256, kernel=(3, 3), pad=(1, 1), block=6)
    conv7 = _up_block(conv6, down_features[2], num_filter=128, kernel=(3, 3), pad=(1, 1), block=7)
    conv8 = _up_block(conv7, down_features[3], num_filter=64, kernel=(3, 3), pad=(1, 1), block=8)
    conv9 = _up_block(conv8, down_features[4], num_filter=32, kernel=(3, 3), pad=(1, 1), block=9)
    conv10 = mx.sym.Convolution(conv9, num_filter=num_classes, kernel=(1, 1), name='conv10_1')
    return conv10


def build_unet(num_classes, inference=False):
    data = mx.sym.Variable(name='data')
    down_features = _down_branch(data)
    decoded = _up_branch(down_features, num_classes)
    channel_softmax = mx.sym.sigmoid(decoded, axis=1)
    if inference:
        return channel_softmax
    else:
        label = mx.sym.Variable(name='label')
        loss = mx.sym.MakeLoss(
            avg_dice_coef_loss(label, channel_softmax),
            normalization='batch'
        )
        mask_output = mx.sym.BlockGrad(channel_softmax, 'mask')
        out = mx.sym.Group([mask_output, loss])
        return out
# Loss and metric derived from Sørensen–Dice_coefficient as described at below: 
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

def avg_dice_coef_loss(y_true, y_pred):
    """
    Symbol for computing weighted average dice coefficient loss

    Parameters
    ----------
    y_true : symbol
        Symbol representing ground truth mask.
    y_pred : symbol
        Symbol representing predicted mask.
    class_weights : symbol
        Symbol of class weights.
    """
    
    intersection = mx.sym.sum(y_true * y_pred, axis=(2, 3))
    numerator = 2. * intersection
    denominator = mx.sym.broadcast_add(mx.sym.sum(y_true, axis=(2, 3)),
                                       mx.sym.sum(y_pred, axis=(2, 3)))
    scores = 1 - mx.sym.broadcast_div(numerator + 1., denominator + 1.)
    
    return mx.sym.mean(scores) 

def avg_dice_coef_metric(y_true, y_pred):
    """
    Method for computing average dice coefficient metric,
    

    Parameters
    ----------
    y_true : array
        Array representing ground truth mask.
    y_pred : array
        Array representing predicted mask.
    num_classes : int
        Number of classes.
    """
    smooth = 1.

    iflat = y_pred.flatten()
    tflat = y_true.flatten()
    intersection = (iflat * tflat).sum()
    
    score = 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
    return score



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=float, default=100)

    parser.add_argument('--root_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=os.environ['SM_HOSTS'])
    return parser.parse_args()
    
if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
    args = parse_args()
    train(args)
    
    

