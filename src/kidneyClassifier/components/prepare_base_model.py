import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.kidneyClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, dropout_rate):
       
        full_model = tf.keras.models.Sequential([
            model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                units=256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                units=classes,
                activation="softmax"
            )
        ])

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,  
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            dropout_rate=self.config.params_dropout_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)