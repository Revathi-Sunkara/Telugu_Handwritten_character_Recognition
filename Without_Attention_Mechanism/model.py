import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
    Reshape, Bidirectional, GRU, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

# ---------------- Build Model ---------------- #
def build_model(input_shape=(64, 64, 1), num_classes=47):
    inputs = Input(shape=input_shape)

    # Convolutional base
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape for RNN
    shape = x.shape
    x = Reshape((shape[1]*shape[2], shape[3]))(x)

    # Bidirectional GRU
    x = Bidirectional(GRU(64, return_sequences=True))(x)

    # Replace attention with Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model