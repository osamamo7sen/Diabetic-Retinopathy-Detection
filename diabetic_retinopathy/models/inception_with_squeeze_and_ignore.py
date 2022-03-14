import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16 as PTModel
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from tensorflow.keras.applications.inception_v3 import InceptionV3 as PTModel
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from tensorflow.keras.models import Model

def inception_with_squeeze_and_ignore(input_shape,num_classes,trainable_backbone = False):

    in_lay = Input(input_shape)
    base_pretrained_model = PTModel(input_shape =  input_shape, include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = trainable_backbone
    pt_depth = 2048
    pt_features = base_pretrained_model(in_lay)

    from tensorflow.keras.layers import BatchNormalization
    # bn_features = BatchNormalization()(pt_features)
    bn_features = pt_features

    # here we do the squeeze and ignore mechanism to turn pixels in the GAP on an off

    sq_ig_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    sq_ig_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(sq_ig_layer)
    sq_ig_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(sq_ig_layer)
    sq_ig_layer = Conv2D(1, 
                        kernel_size = (1,1), 
                        padding = 'valid', 
                        activation = 'sigmoid')(sq_ig_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    sq_ig_layer = up_c2(sq_ig_layer)

    mask_features = multiply([sq_ig_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(sq_ig_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    out_layer = Dense(num_classes)(dr_steps)
    retina_model = Model(inputs = [in_lay], outputs = [out_layer])
    return retina_model
