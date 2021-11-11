>author Stephanie
>
>reference https://github.com/weiaicunzai/pytorch-cifar100

### Implemented model: ResNet, DenseNet, ShuffleNet

*All implementation based on corresponding model paper. It is suitable as a reference for reading papers.*

#### About Modify to suit each model or to tune

To modify options such as is_gpu and is_resume or super parameters such as LR, please modify config.py`

Some strategy changes need to modify the source code of train.py (e.g. modify the learning rate schedule)

`PS:ShuffleNet accuracy is not good, welcome to practice model tuning.`