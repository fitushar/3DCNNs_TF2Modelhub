import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import *



def VGG3D(inputs,num_classes):
    inputs = inputs
    x = inputs


    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling3D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=num_classes,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    return model
