import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter

from kidneyClassifier.entity.config_entity import TrainingConfig
tf.config.run_functions_eagerly(True)



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    
    def get_base_model(self):
        # Load the model WITHOUT compiling (this avoids the optimizer error)
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False  # KEY FIX: Don't load the old optimizer
        )
        
        # Recompile with fresh optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        print(f"✓ Model loaded and recompiled with learning rate: {self.config.params_learning_rate}")
    
    
    def train_valid_generator(self):
        """
        Creates training and validation generators from a single directory.
        Uses validation_split=0.20 to automatically split data into 80% train and 20% validation.
        """
        
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'
        )

        # Validation generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator (with augmentation if enabled)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )


    def compute_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        # Get class labels
        labels = self.train_generator.classes
        
        # Count samples per class
        class_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # Compute balanced class weights: total / (num_classes * count_per_class)
        class_weight_dict = {}
        for class_idx, count in class_counts.items():
            class_weight_dict[class_idx] = total_samples / (num_classes * count)
        
        print("\nClass weights computed:")
        for class_name, class_idx in self.train_generator.class_indices.items():
            print(f"  {class_name} (samples: {class_counts[class_idx]}): weight = {class_weight_dict[class_idx]:.4f}")
        
        return class_weight_dict


    def get_callbacks(self):
        """Create callbacks for training"""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.params_early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.params_reduce_lr_factor,
            patience=self.config.params_reduce_lr_patience,
            min_lr=self.config.params_min_lr,
            verbose=1
        )

        return [early_stopping, reduce_lr]


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Compute class weights
        class_weight_dict = self.compute_class_weights()
        
        # Get callbacks
        callbacks = self.get_callbacks()

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )