# PAD-Net-PyTorch

Thanks for your attention. In this repo, we provide the codes for the paper [[Binocular Rivalry Oriented Predictive Auto-Encoding Network for Blind Stereoscopic Image Quality Measurement]](https://arxiv.org/abs/1909.01738).

## Prerequisites
+ scipy==1.2.1
+ opencv_python==4.1.0.25
+ numpy==1.16.4
+ torchvision==0.3.0
+ torch==1.1.0
+ Pillow==6.2.0

## Install
To install all the dependencies in prerequisites

## Prepare Data
+ Obtain [pretrained_weight.pth](https://drive.google.com/file/d/1xsRaJOPbLVG58Fco7tCk92v0NBngCUhP/view?usp=sharing), [live1_model.pth](https://drive.google.com/file/d/1fZ3cXUW0ueTzecSp5eqJm_22UJWzKAmV/view?usp=sharing), and [live2_model.pth](https://drive.google.com/file/d/1cgWsmUclN54rvwAHyNwriqqrHouyStg4/view?usp=sharing)
+ Download [database](https://drive.google.com/drive/folders/1QkhhfFlaNu6v0jhumtgxQFD3TcVECA3S?usp=sharing)

## Training
```
python main_live2.py --save model_live1 --total_epochs 300 --lr 1e-4 --batch_size 16 --log_interval 5 --resume pretrained_weight.pth
```

## Testing
```
python main_live2.py --skip_training --resume live2_model.pth
```

## Citation
You may cite it in your paper. Thanks a lot.

```
@article{xu2020binocular,
  title={Binocular Rivalry Oriented Predictive Autoencoding Network for Blind Stereoscopic Image Quality Measurement},
  author={Xu, Jiahua and Zhou, Wei and Chen, Zhibo and Ling, Suiyi and Le Callet, Patrick},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={70},
  pages={1--13},
  year={2020},
  publisher={IEEE}
}
```


