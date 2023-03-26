# Project for UM classes
Testing various CNN models for image classification

## Dataset source:
https://www.kaggle.com/datasets/alessiocorrado99/animals10


## Tested models
1. Tested preprocessing parameters for resnet18, weight initialized, 12 min training
{
    20221209_195344: 'RandomCropResize, RandomFlip, Normalization'
    20221209_205319: 'RandomCropResize, RandomFlip'
}

2. Tested preprocessing parameters for resnet18, weights RANDOM, 12 min training
{
    20221209_211433: 'RandomCropResize, RandomFlip, Normalization'
}

3. AlexNet, weights RANDOM, 7.5 min training
{
    20221209_213015: 'RandomCropResize, RandomFlip, Normalization', 
}

4. AlexNet, weights imagenet, 7.5 min training
{
    20221209_215731: 'RandomCropResize, RandomFlip, Normalization', 
}
on w sumie zadziala nawet bez pretrained

5. Lenet, weights RANDOM, 12 min training
{
    20221209_230012: 'RandomCropResize, RandomFlip, Normalization', 
}
