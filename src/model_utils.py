from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model,Input,regularizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

 # Create model configration from where we extract the model and it
model_configs = {
    "vgg16": {
        "model_class": VGG16,
        "preprocess_fn": vgg16_preprocess,
        "input_shape": (224, 224, 3)
    },
    "vgg19": {
        "model_class": VGG19,
        "preprocess_fn": vgg19_preprocess,
        "input_shape": (224, 224, 3)
    },
    "inceptionv3": {
        "model_class": InceptionV3,
        "preprocess_fn": inception_preprocess,
        "input_shape": (299, 299, 3)  # Inception requires larger input
    }
}



def setup_datagenerator(train_dir, val_dir, test_dir, preprocess_fn, batch_size=16, image_size=(224, 224)):
    """
    Create and configure data generators for the model.
    
    Parameters:
    - train_dir: Path to training data directory
    - val_dir: Path to validation data directory
    - test_dir: Path to test data directory
    - preprocess_fn: Preprocessing function to apply to images
    - batch_size: Batch size for generators (default: 32)
    - image_size: Target image size as (height, width) (default: (224, 224))
    
    Returns:
    - train_generator: Training data generator
    - validation_generator: Validation data generator
    - test_generator: Test data generator
    """
    # Create data generators with appropriate preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )
    
    # Validation and test generators don't need rescaling if preprocess_fn handles it
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

 # Build the model  
def build_model(base_model_class, preprocess_fn, num_classes, input_shape=(224, 224, 3)):
    """Create a transfer learning model with the given base model"""
    # Create the base model
    base_model = base_model_class(
        include_top=False, 
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    

    inputs = Input(shape=input_shape)
    x = preprocess_fn(inputs)
    x = base_model(x, training=False) # Keep False to freeze base model initially

# Feature extraction and regularization head
    x = GlobalAveragePooling2D()(x)

# Add BatchNormalization to stabilize and accelerate training
    x = BatchNormalization()(x)

# Use a smaller, regularized Dense layer
    x = Dense(256, activation='relu', # 256 is often sufficient
                 kernel_regularizer=regularizers.l2(1e-4))(x) # Small L2 penalty
    x = Dropout(0.5)(x)

# Optional: Second smaller Dense layer for more capacity
    x = Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x) # Slightly lower dropout

# Final output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Train the model 
def train_model(model_name, config, train_gen, val_gen,num_classes, epochs=50):
    """Train the specified model using provided generators"""
    print(f"\nTraining model {model_name}.....")
    
    # Build the model
    model = build_model(
        config['model_class'],
        config['preprocess_fn'],
        num_classes,
        config['input_shape']
    )
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        f'{model_name}_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(log_dir=f'logs/experiment_N1/{model_name}')
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    return model, history