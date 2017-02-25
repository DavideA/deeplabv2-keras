# deeplabv2-keras
This repository holds a keras porting of the deeplabV2 model presented
in [this paper](https://arxiv.org/pdf/1606.00915v1.pdf).

**Please note that only the VGG encoder version with voc pretraining
on VOC12 segmentation is held in this repo**.

The model is developed and tested with Theano backend. In order to get
it working with Tensorflow, you'll have to code the `BilinearUpsampling`
layer, and convert all weights.

Moreover, this network does not include the CRF post processing,
but it can be integrated with [this code](https://github.com/lucasb-eyer/pydensecrf).

## How to use
Simply get the model with the function `DeeplabV2` in `deeplabv2.py`.
For an example, check `predict.py`.

##### TODOS:
* add model with ResNet-101 encoder.
