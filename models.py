from xml.etree.ElementInclude import include
from .blocks import (
    decoder_block,
    conv_bn_block,
    transpose_bn_block,
    AtrousSpatialPyramidPooling,
    aspp_block,
)
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate


def Unet(
    input_shape,
    backbone="efficientnetb3",
    classes=1,
    decoder_activation="relu",
    final_activation="sigmoid",
    filters=[256, 128, 64, 32, 16],
):

    if backbone == "efficientnetb3":

        encoder = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ]
        x = encoder.get_layer("top_activation").output

    elif backbone == "mobilenetv2":

        encoder = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block_13_expand_relu",
            "block_6_expand_relu",
            "block_3_expand_relu",
            "block_1_expand_relu",
        ]
        x = encoder.get_layer("out_relu").output

    elif backbone == "resnet50":

        encoder = tf.keras.applications.resnet50.ResNet50(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "conv4_block6_out",
            "conv3_block4_out",
            "conv2_block3_out",
            "conv1_relu",
        ]
        x = encoder.get_layer("conv5_block3_out").output

    elif backbone == "efficientnetv2_b3":
        encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block2c_add",
            "block1b_add",
        ]
        x = encoder.get_layer("top_activation").output

    else:
        list_of_backbones = [
            "efficientnetb3",
            "mobilenetv2",
            "resnet50",
            "efficientnetv2_b3",
        ]
        raise ValueError(f"Valid backbones are: {list_of_backbones}")

    skip_connections = []
    for layer in layers:
        skip_connections.append(encoder.get_layer(layer).output)

    for i, skip in enumerate(skip_connections):
        if backbone == "efficientnetv2_b3" and i > 1:
            skip = BatchNormalization(axis=-1, name=f"BatchNorm_{i}")(skip)
            skip = Activation("swish", name=f"Activation_{i}")(skip)
        x = decoder_block(
            inputs=x,
            filters=filters[i],
            stage=i,
            skip=skip,
            activation=decoder_activation,
        )

    x = decoder_block(
        inputs=x, filters=filters[-1], stage=4, activation=decoder_activation
    )

    x = Conv2D(filters=classes, kernel_size=(3, 3), padding="same", name="final_conv")(
        x
    )
    x = Activation(final_activation, name=final_activation)(x)

    model = tf.keras.models.Model(encoder.input, x)

    return model


def ASPP_Unet(
    input_shape,
    backbone="efficientnetb3",
    classes=1,
    decoder_activation="relu",
    final_activation="sigmoid",
    filters=[256, 128, 16],
):

    if backbone == "efficientnetb7":

        encoder = tf.keras.applications.efficientnet.EfficientNetB7(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block3a_expand_activation",
            "block2a_expand_activation",
        ]
        x = encoder.get_layer("block4a_expand_activation").output

    elif backbone == "resnet50":

        encoder = tf.keras.applications.resnet50.ResNet50(
            include_top=False, input_shape=input_shape, include_preprocessing=False
        )
        layers = [
            "conv2_block3_out",
            "conv1_relu",
        ]
        x = encoder.get_layer("conv3_block4_out").output

    elif backbone == "resnet152":
        encoder = tf.keras.applications.resnet.ResNet152(
            include_top=False, input_shape=input_shape, include_preprocessing=False
        )
        layers = [
            "conv2_block3_out",
            "conv1_relu",
        ]
        x = encoder.get_layer("conv3_block8_out").output

    else:
        list_of_backbones = [
            "efficientnetb7",
            "resnet50",
            "resnet152",
        ]
        raise ValueError(f"Valid backbones are: {list_of_backbones}")

    skip_connections = []
    for layer in layers:
        skip_connections.append(encoder.get_layer(layer).output)

    x = aspp_block(x, 256)

    for i, skip in enumerate(skip_connections):
        if backbone == "efficientnetv2_b3" and i > 1:
            skip = BatchNormalization(axis=-1, name=f"BatchNorm_{i}")(skip)
            skip = Activation("swish", name=f"Activation_{i}")(skip)
        x = decoder_block(
            inputs=x,
            filters=filters[i],
            stage=i,
            skip=skip,
            activation=decoder_activation,
        )

    x = decoder_block(
        inputs=x, filters=filters[-1], stage=4, activation=decoder_activation
    )

    x = Conv2D(filters=classes, kernel_size=(3, 3), padding="same", name="final_conv")(
        x
    )
    x = Activation(final_activation, name=final_activation)(x)

    model = tf.keras.models.Model(encoder.input, x)

    return model


def DeepLabV3(
    image_size,
    classes,
    final_activation="sigmoid",
    dilation_rates=(1, 6, 12, 18),
    backbone="resnet101",
):

    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    if backbone == "resnet101":
        encoder = tf.keras.applications.resnet.ResNet101(
            input_tensor=model_input, include_top=False, weights="imagenet"
        )

        layers = ["conv4_block6_2_relu", "conv2_block3_2_relu"]

    elif backbone == "efficientnetb3":
        encoder = tf.keras.applications.efficientnet.EfficientNetB3(
            input_tensor=model_input, include_top=False
        )
        layers = ["block6a_expand_activation", "block3a_expand_activation"]

    else:
        backbone_list = ["resnet101", "efficientnetb3"]
        raise ValueError(f"Valid backbones are: {backbone_list}")

    x = encoder.get_layer(layers[0]).output
    x = AtrousSpatialPyramidPooling(x, dilation_rates=dilation_rates)

    input_a = UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = encoder.get_layer(layers[1]).output
    input_b = conv_bn_block(input_b, filters=48, k_size=1, name="decoder_stage0")

    x = Concatenate(axis=-1)([input_a, input_b])
    x = conv_bn_block(x, filters=256, k_size=3, name="decoder_stage1_0")
    x = conv_bn_block(x, filters=256, k_size=3, name="decoder_stage1_1")
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(
        classes, kernel_size=1, activation=final_activation, padding="same"
    )(x)

    return tf.keras.Model(inputs=model_input, outputs=model_output)
