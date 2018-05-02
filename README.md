# tf-mobilenet-v2
Mobilenet V2(Inverted Residual) Implementation &amp; Trained Weights Using Tensorflow

## Training

```
$ python3 run.py train --num_gpu=1 --depth_multiplier=1.0 --datadir=... 
```

## Pretrained Models

I trained in a few ways, but I failed to replicate the result from the original paper. (2~4% Accuracy Drop)

But you can use the pretrained weight in tensorflow now : https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

## References

- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
- [Paper Review (Korean)](http://openresearch.ai/t/mobilenetv2-inverted-residuals-and-linear-bottlenecks-mobile-networks-for-classification-detection-and-segmentation/130/1)
- [Mobilenet V2 Tensorflow Implementation : https://github.com/timctho/mobilenet-v2-tensorflow](https://github.com/timctho/mobilenet-v2-tensorflow)
- [Mobilenet V2 Pytorch Implementation : https://github.com/MG2033/MobileNet-V2](https://github.com/MG2033/MobileNet-V2)
- [Mobilenet V1 Official Implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)
- [Mobilenet V1 Official Test](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_test.py)
- [Checkmate : Tensorflow Drop-in Saver](https://github.com/vonclites/checkmate)
