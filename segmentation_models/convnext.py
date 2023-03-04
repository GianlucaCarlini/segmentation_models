from .blocks import ConvNextBlock, DropPath
import tensorflow as tf


def ConvNext(
    input_shape=(256, 256, 3),
    num_classes=1000,
    depths=[3, 3, 9, 3],
    dims=96,
    drop_path_rate=0.0,
    layer_scale_init_value=None,
    model_name="ConvNext",
    patch_size=4,
    include_top=True,
):

    num_layers = len(depths)

    dpr = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

    input = tf.keras.Input(shape=input_shape, name="Input")

    if not isinstance(dims, list):
        dims_list = [dims * (2**i) for i in range(num_layers)]
        dims = dims_list

    x = tf.keras.layers.Conv2D(
        filters=dims[0],
        kernel_size=patch_size,
        strides=patch_size,
        name="PatchEmbedding",
    )(input)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="first_LayerNorm")(x)

    dpr_index = 0
    for i in range(num_layers):
        for d in range(depths[i]):
            x = ConvNextBlock(
                dim=dims[i],
                drop_path=dpr[dpr_index + d],
                layer_scale_init_value=layer_scale_init_value,
                name=f"stage_{i}_ConvNextBlock_{d}",
            )(x)

        if i < num_layers - 1:
            x = tf.keras.layers.Conv2D(
                filters=dims[i + 1],
                kernel_size=2,
                strides=2,
                name=f"stage_{i}_DownSample",
            )(x)
            x = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"stage_{i}_LayerNorm"
            )(x)

        dpr_index += depths[i]

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"final_LayerNorm")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name=f"final_GlobalAveragePooling")(x)

    if include_top:
        out = tf.keras.layers.Dense(num_classes, name="head")(out)

    model = tf.keras.models.Model(inputs=input, outputs=out, name=model_name)

    return model


def UNext(
    input_shape=(256, 256, 3),
    num_classes=3,
    depths=[3, 3, 9, 3],
    dims=96,
    drop_path_rate=0.0,
    layer_scale_init_value=None,
    model_name="ConvNextUnet",
    patch_size=4,
    final_activation="sigmoid",
):

    num_layers = len(depths)

    if not isinstance(dims, list):
        dims_list = [dims * (2**i) for i in range(num_layers)]
        dims = dims_list

    encoder = ConvNext(
        input_shape=input_shape,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        patch_size=patch_size,
    )

    blk = [i - 1 for i in depths]
    stage = [i for i in range(len(depths))]

    layers = [f"stage_{s}_ConvNextBlock_{b}" for s, b in zip(stage, blk)]
    layers.reverse()

    skip_connections = []

    for layer in layers[1:]:
        skip_connections.append(encoder.get_layer(layer).output)

    dims.reverse()

    x = encoder.get_layer(layers[0]).output

    for i, skip in enumerate(skip_connections):

        x = tf.keras.layers.Dense(units=dims[i + 1], name=f"decoder_stage_{i}_embed_1")(
            x
        )
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bicubic", name=f"decoder_stage_{i}_Upsampling2D"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"decoder_stage_{i}_Concatenate")(
            [x, skip]
        )
        x = tf.keras.layers.Dense(units=dims[i + 1], name=f"decoder_stage_{i}_embed_2")(
            x
        )
        x = ConvNextBlock(
            dim=dims[i + 1],
            drop_path=None,
            layer_scale_init_value=layer_scale_init_value,
            name=f"decoder_stage_{i}_ConvNext_block_1",
        )(x)
        x = ConvNextBlock(
            dim=dims[i + 1],
            drop_path=None,
            layer_scale_init_value=layer_scale_init_value,
            name=f"decoder_stage_{i}_ConvNext_block_2",
        )(x)

    x = tf.keras.layers.UpSampling2D(
        size=patch_size, interpolation="bicubic", name=f"decoder_final_Upsampling2D"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=dims[-1] // 2,
        padding="same",
        kernel_size=3,
        activation="gelu",
        name="decoder_final_Conv2D",
    )(x)
    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="decoder_final_LayerNormalization"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation=final_activation,
        name="decoder_output",
    )(x)

    model = tf.keras.models.Model(inputs=encoder.input, outputs=x, name=model_name)

    return model
