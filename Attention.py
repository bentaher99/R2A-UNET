import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, 
                                   Multiply, GlobalMaxPooling2D, Concatenate,
                                   GlobalAveragePooling2D, Dense, Reshape)

class SpatialAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        
        self.concat = Concatenate()
        self.conv = Conv2D(1, kernel_size=3, padding="same")
        self.batch_norm = BatchNormalization()
        self.activation = Activation('sigmoid')
    
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        attention = self.conv(concat)
        attention = self.batch_norm(attention)
        attention = self.activation(attention)
        return Multiply()([inputs, attention])

def Attention_mechanism(x):
    channel = x.shape[-1]
    
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    channel_attention = avg_pool + max_pool
    channel_attention = Dense(channel, use_bias=False)(channel_attention)
    channel_attention = BatchNormalization()(channel_attention)
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    
    feats_channel = Multiply()([x, channel_attention])
    feats_spatial = SpatialAttentionLayer()(x)
    
    return feats_spatial, feats_channel

def attention_block(inputs, skip_inputs, num_filters):
    g = Conv2D(num_filters, 1, use_bias=False, kernel_initializer="he_normal")(skip_inputs)
    x = Conv2D(num_filters, 1, use_bias=False, kernel_initializer="he_normal")(inputs)
    theta = Conv2D(num_filters, 1, use_bias=False, kernel_initializer="he_normal")(inputs)
    phi = Conv2D(num_filters, 1, use_bias=False, kernel_initializer="he_normal")(skip_inputs)
    
    theta_x = tf.keras.layers.multiply([theta, g])
    phi_g = tf.keras.layers.multiply([phi, x])
    concat = concatenate([theta_x, phi_g])
    concat = Conv2D(num_filters, 1, use_bias=False, kernel_initializer="he_normal")(concat)
    concat = Activation("relu")(concat)
    
    return concat