import tensorflow as tf
import numpy as np
from .blocks import PatchExpanding, PatchEmbed, PatchMerging
from .blocks import SwinBasicLayer, SwinDecoderLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D


def SwinTransformer(
    input_shape=(224, 224, 3),
    model_name="swin_tiny_patch4_window7_224",
    include_top=False,
    patch_size=(4, 4),
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=LayerNormalization,
    ape=False,
    patch_norm=True,
    **kwargs,
):

    if input_shape[0] % window_size != 0 or input_shape[1] % window_size != 0:
        raise ValueError(
            (
                f"Image input dimensions must be a multiple of window_size. "
                f"Image dimensions of ({input_shape[0]} X {input_shape[1]}) and window_size of {window_size} were given"
            )
        )

    img_size = (input_shape[0], input_shape[1])
    in_chans = input_shape[2]

    dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer if patch_norm else None,
    )
    num_patches = patch_embed.num_patch

    dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]

    num_layers = len(depths)
    patches_resolution = patch_embed.patch_resolution

    input = tf.keras.Input(shape=input_shape)
    x = patch_embed(input)

    x = Dropout(drop_rate)(x)

    for i_layer in range(num_layers):

        x = SwinBasicLayer(
            x,
            dim=int(embed_dim * 2**i_layer),
            input_resolution=(
                patches_resolution[0] // (2**i_layer),
                patches_resolution[1] // (2**i_layer),
            ),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_prob=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer < num_layers - 1) else None,
            prefix=f"stage_{i_layer}",
        )

    x = LayerNormalization(epsilon=1e-5)(x)
    out = GlobalAveragePooling1D()(x)
    if include_top:
        out = Dense(num_classes, name="head")(out)

    swin_transformer = tf.keras.models.Model(inputs=input, outputs=out, name=model_name)

    return swin_transformer


def Swin_Unet(
    input_shape=(224, 224, 3),
    model_name="swin_tiny_224",
    classes=1,
    final_activation="sigmoid",
    patch_size=(4, 4),
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=LayerNormalization,
    ape=False,
    patch_norm=True,
    **kwargs,
):

    output_stride = 2 ** (len(depths) + 1)

    if input_shape[0] % output_stride != 0 or input_shape[1] % output_stride != 0:
        raise ValueError(
            (
                f"Input shape must be a multiple of output stride (output stride = 2 ** (len(depth) + 1)). "
                f"Input shape = ({input_shape[0]} x {input_shape[1]}) and output stride = {output_stride} were given"
            )
        )

    encoder = SwinTransformer(
        input_shape=input_shape,
        model_name=model_name,
        include_top=False,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        ape=ape,
        patch_norm=patch_norm,
    )

    input_resolution = (
        input_shape[0] // output_stride,
        input_shape[1] // output_stride,
    )

    blk = [i - 1 for i in depths]
    stage = [i for i in range(len(depths))]

    layers = [f"stage_{s}_block_{b}" for s, b in zip(stage, blk)]
    layers.reverse()

    skip_connections = []

    for layer in layers[1:]:

        skip_connections.append(encoder.get_layer(layer).output)

    x = encoder.get_layer(layers[0]).output

    for i, skip in enumerate(skip_connections):

        x = SwinDecoderLayer(
            input=x,
            skip=skip,
            dim=int(embed_dim * 2 ** (len(depths) - 1 - i)),
            input_resolution=(
                input_resolution[0] * (2**i),
                input_resolution[1] * (2**i),
            ),
            depth=2,
            num_heads=num_heads[-(i + 1)],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_prob=0.0,
            norm_layer=norm_layer,
            expand_dims=True if i == 0 else False,
            upsample=PatchExpanding,
            prefix=f"decoder_{i}",
        )

    x = PatchExpanding(
        input_resolution=(
            input_resolution[0] * (output_stride // 4),
            input_resolution[1] * (output_stride // 4),
        ),
        dim=x.shape[-1],
        expand_dims=True,
        upsample=4,
        prefix="decoder_final",
    )(x)

    x = tf.reshape(x, shape=(-1, input_shape[0], input_shape[1], x.shape[-1]))

    x = Conv2D(classes, kernel_size=1, activation=final_activation, name="final_conv")(
        x
    )

    swin_unet = tf.keras.models.Model(inputs=encoder.input, outputs=x, name="Swin Unet")

    return swin_unet
