# How to run the code

* Python version: 3.8.5
* Install the requirements:
> $ pip install -r requirments.txt

## Diabetic Retinopathy
### To train a new model from scratch:
* First specify the model type in config.gin
  * vgg_like
  * inception_with_squeeze_and_ignore


* In the command line:
> $ python3 main.py --train

### To continue training:
* In the command line:
> $ python3 main.py --train --checkpoint /path/to/Checkpoints

### For transfer learning:
* In the comman line:
> $ python3 main.py --train --transfer_model /path/to/saved/model

### To evaluate a model:

* In the command line:
>$ python3 main.py --checkpoint /path/to/Checkpoints/directory

  >$ python3 main.py --saved_model /path/to/saved/model

### Notes:
* The visualization is logged into the tensorboard.
* To run tensorboard:
>$ tensorboard --logdir /path/to/directory/of/the/summary

## Human Activity Recognition
### First prepare the dataset:
* In the config.gin:
  * prepare_HAPT.dataset_dir: the directory of the original dataset.
  * prepare_HAPT.out_dir: the directory to store the tf_records.
  * load.data_dir: the parent directoryof the "prepare_HAPT.out_dir" directory.
* In the command line:
>$ python3 main.py --prepare

### To train a new model:
* First specify the model type in config.gin
  * lstm_double
  * cnn_lstm


* In the command line:
> $ python3 main.py --train

### To continue training:
* In the command line:
> $ python3 main.py --train --checkpoint /path/to/Checkpoints

  > $ python3 main.py --train --saved_model /path/to/saved/model

### To evaluate a model:

* In the command line:
>$ python3 main.py --checkpoint /path/to/Checkpoints/directory

  >$ python3 main.py --saved_model /path/to/saved/model

### Notes:
* The visualization is logged into the tensorboard.
* To run tensorboard:
>$ tensorboard --logdir /path/to/directory/of/the/summary

# Results

## Diabetic Retinopathy

| Model(arch\_weights\_dataset\_preprocess)        | Balanced Accuracy             |
| ------------- |:-------------:|
| VGG-like_None_IDRID_None      | 63.2% |
| ISIC_ImgNet_IDRID_None      | 79.9%      |
| ISIC_ImgNet_IDRID_BTGraham | 81.0%      |
| ISIC_None_EyePacs_BTGraham | 83.5%      |
| ISIC_EyePacs_IDRID_BTGraham | 80.3%      |
| + SIC tranied | 85.6%     |
| + transfer learning | 87.5%      |

## Human Activity Recognition
The model achieves 94.3\% classification accuracy and 80.9\% balanced accuracy on the test set. The drop in the balanced accuracy is because of the underrepresented transitional classes. the model mixes between setting and standing.
