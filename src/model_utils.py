# model_utils.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_datagenerator(train_dir, val_dir, test_dir, preprocess_fn, batch_size=32, image_size=(224, 224)):
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
        rescale=1./255,
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