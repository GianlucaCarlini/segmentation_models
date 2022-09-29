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
    model_name="swin_tiny_224",
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
    """Instantiates the Swin Transformer Model. The default arguments are the ones of the
       original Swin Transformer tiny. The implementation is slightly different from the original
       since it uses the SpaceToDepth tf layer to perform the patch merging operation. It is indeed the
       same operation of the original PatchMerging definded by the authors, but I preferred to use
       a standard tf layer rather than the custom implementation of the authors.
       Probably this does not allow to use the original pre-trained weights, but I still have
       to test it.


    Args:
        input_shape (tuple, optional): The input shape of the model. Defaults to (224, 224, 3).
        model_name (str, optional): The model name. Defaults to "swin_tiny_224".
        include_top (bool, optional): Whether to include or not the top of the encoder. Defaults to False.
        patch_size (tuple, optional): The size of the patches for the embedding. Defaults to (4, 4).
        num_classes (int, optional): The number of classes to predict, only meaningful if include top is True. Defaults to 1000.
        embed_dim (int, optional): The initial embedding dimension of the SwinBlock. Defaults to 96.
        depths (list, optional): The number of SwinTransformer layers, each composed by depths[i] blocks. Defaults to [2, 2, 6, 2].
        num_heads (list, optional): The number of heads in the multi-head self-attention for each layer. Defaults to [3, 6, 12, 24].
        window_size (int, optional): The size of the attention window. Defaults to 7.
        mlp_ratio (float, optional): The ratio of hidden neurons to input neurons in the Attention layer. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to add the bias in the qkv calculation. Defaults to True.
        qk_scale (_type_, optional): Scale of the qk attention matrix, if None, sqrt(dim/num_heads) is used. Defaults to None.
        drop_rate (float, optional): The dropout rate. Defaults to 0.0.
        attn_drop_rate (float, optional): The attention dropout rate. Defaults to 0.0.
        drop_path_rate (float, optional): The drop path rate. Defaults to 0.1.
        norm_layer (_type_, optional): The normalization layer. Defaults to LayerNormalization.
        ape (bool, optional): Does nothing, to be implemented. Defaults to False.
        patch_norm (bool, optional): Whether to normalize or not the initial patch embedding. Defaults to True.

    Returns:
        tf.Model: The builded SwinTransformer Model
    """

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
    """Instantiates a UNet-like model with a Swin Transformer backbone.
    The decoder has the same layer types of the encoder, meaning that the whole
    model only uses the mechanism of attention (no convolution is used, except
    for the output). The upsampling also follows the Swin Transformer implementation
    and uses the tensorflow layer depth to space.

    Args:
        input_shape (tuple, optional): The shape of the input tensor. Defaults to (224, 224, 3).
        model_name (str, optional): The name of the model. Defaults to "swin_tiny_224".
        classes (int, optional): The number of output classes, i.e., the number of
            output feature maps Defaults to 1.
        final_activation (str, optional): The activation function of the output layer.
            Defaults to "sigmoid".
        patch_size (tuple, optional): The size of the patches for the embedding.
            Defaults to (4, 4).
        embed_dim (int, optional): The initial embedding dimension of the SwinBlock. Defaults to 96.
        depths (list, optional): The number of SwinTransformer layers, each composed by depths[i] blocks.
            Defaults to [2, 2, 6, 2].
        num_heads (list, optional): The number of heads in the multi-head self-attention for each layer.
            Defaults to [3, 6, 12, 24].
        window_size (int, optional): The size of the attention window. Defaults to 7.
        mlp_ratio (float, optional): The ratio of hidden neurons to input neurons in the Attention layer.
            Defaults to 4.0.
        qkv_bias (bool, optional): Whether to add the bias in the qkv calculation. Defaults to True.
        qk_scale (_type_, optional): Scale of the qk attention matrix, if None, sqrt(dim/num_heads) is used.
            Defaults to None.
        drop_rate (float, optional): The dropout rate. Defaults to 0.0.
        attn_drop_rate (float, optional):The attention dropout rate. Defaults to 0.0.
        drop_path_rate (float, optional): The drop path rate. Defaults to 0.1.
        norm_layer (_type_, optional): The normalization layer. Defaults to LayerNormalization.
        ape (bool, optional): Does nothing, to be implemented. Defaults to False.
        patch_norm (bool, optional): Whether to normalize or not the initial patch embedding. Defaults to True.

    Returns:
        tf.Model: The builded SwinTransformer UNet Model
    """

    output_stride = (2 ** len(depths)) * (patch_size[0] // 2)

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
            input_resolution[0] * (output_stride // patch_size[0]),
            input_resolution[1] * (output_stride // patch_size[1]),
        ),
        dim=x.shape[-1],
        expand_dims=True,
        upsample=patch_size[0],
        prefix="decoder_final",
    )(x)

    x = tf.reshape(x, shape=(-1, input_shape[0], input_shape[1], x.shape[-1]))

    x = Conv2D(classes, kernel_size=1, activation=final_activation, name="final_conv")(
        x
    )

    swin_unet = tf.keras.models.Model(inputs=encoder.input, outputs=x, name=model_name)

    return swin_unet
