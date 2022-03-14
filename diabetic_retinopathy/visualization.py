import logging
import io

import gin
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from input_pipeline import datasets
from models.architectures import vgg_like, resnet
from utils import utils_params, utils_misc
from models.architectures import build_model

"""
Code is based on the example by tensorflow: https://www.tensorflow.org/tutorials/interpretability/integrated_gradients

"""

def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


@tf.function
def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         model,
                         m_steps=50,
                         batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx,
                                           model=model)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

        # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def plot_img_attributions(baseline,
                          image,
                          vis_image,
                          target_class_idx,
                          label,
                          model,
                          imageName,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline,
                                        image=image,
                                        target_class_idx=target_class_idx,
                                        m_steps=m_steps,
                                        model=model)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))

    #axs[0, 0].set_title('Baseline image')
    #axs[0, 0].imshow(baseline)
    #axs[0, 0].axis('off')

    axs[0, 0].set_title('Original image')
    axs[0, 0].imshow(vis_image)
    axs[0, 0].axis('off')

    #axs[1, 0].set_title('Attribution mask')
    #axs[1, 0].imshow(attribution_mask, cmap=cmap)
    #axs[1, 0].axis('off')

    axs[0, 1].set_title('Overlay')
    axs[0, 1].imshow(attribution_mask, cmap=cmap)
    axs[0, 1].imshow(vis_image, alpha=overlay_alpha)
    axs[0, 1].axis('off')

    plt.tight_layout()
    title = "T: " + str(tf.reduce_sum(label).numpy()) + ", P: " + str(target_class_idx)
    plt.title(title)
    if imageName != '' :
        plt.savefig(imageName)
    return fig


def visualize():
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    ckpt_path = '/misc/home/RUS_CIP/st170249/inception/experiments/run_2021-02-06T11-34-22-739804/ckpts'
    model = tf.saved_model.load(ckpt_path)

    #Iterate over the dataset
    iterator = iter(ds_test)
    for i in range(50):
        #extact image and label form dataset
        sample_data = iterator.__next__()
        image_sample = sample_data[0]
        print(sample_data[2].shape)
        label_sample = tf.cast(sample_data[1], dtype=tf.int32)
        #plot prediction
        prediction = (tf.argmax(tf.nn.softmax(model(image_sample), axis=-1), axis=-1)).numpy()[0]
        #Image name to be saved
        imageName = "visualized_image_" + str(i + 1)
        baseline = tf.zeros(shape=ds_info['image_shape'])
        #plot the visualization
        _ = plot_img_attributions(image=tf.squeeze(sample_data[0]),
                                  vis_image= tf.squeeze(sample_data[2]),
                                  baseline=baseline,
                                  target_class_idx=prediction,
                                  label=label_sample,
                                  model=model,
                                  imageName=imageName,
                                  m_steps=150,
                                  cmap=plt.cm.inferno,
                                  overlay_alpha=0.4)


#    break
def visualize_integrated_gradients(model,images,labels,images_original):

    #visualize image
    return_images = list()
    for image,label,image_original in zip(images,labels,images_original):
        tf.cast(label,tf.int32)
        prediction = (tf.argmax(tf.nn.softmax(model(tf.expand_dims(image,axis=0)), axis=-1), axis=-1)).numpy()[0]  
        baseline = tf.zeros_like(image)
        result_image = plot_img_attributions(image=tf.squeeze(image),
                            vis_image= tf.squeeze(image_original),
                            baseline=baseline,
                            target_class_idx=prediction,
                            label=label,
                            model=model,
                            imageName='',
                            m_steps=150,
                            cmap=plt.cm.inferno,
                            overlay_alpha=0.4)
        # return_images.append(result_image)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buffer.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image 
        return_images.append(image)

        plt.close(result_image)


    return return_images 



if __name__ == '__main__':
    visualize()
