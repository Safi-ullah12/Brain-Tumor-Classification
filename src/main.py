from Dataset_pipeline import create_dataset_pipeline
from EDA_origanal_preprocess import visualize_dataset_distribution, plot_training_history
from model_utils import setup_datagenerator, model_configs,train_model
from evaluation_utils import evaluate_model, plot_model_comparison
import tensorflow as tf
import gc
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

def main():
    # Define  parameters directly in the script
    source_dir = "/content/drive/MyDrive/MRI_Orignal Data"  # Update with your actual path
    base_dir = "/content/drive/MyDrive/my_project"  # Update with your actual path
    result_dir = "/content/drive/MyDrive/my_project/results"  # Update with your actual path
    image_size = (224, 224)
    val_ratio = 0.10
    batch_size = 16
    data_type = "preprocessed"  # or "original"
    num_classes = 4
    print("\n" + "="*50)
    print("Starting Dataset Visualization")
    print("="*50)
    visualize_dataset_distribution(dataset_dir=source_dir,
                                   result_dir=result_dir,
                                   data_type='original')
    
    print("="*50)
    print("Starting Dataset Processing Pipeline")
    print("="*50)
    
    # Process the dataset
    pipeline_result = create_dataset_pipeline(
        source_dir=source_dir,
        base_dir=base_dir,
        image_size=(224, 224),  # Use standard size for processing
        val_ratio=val_ratio
    )
    
    # Extract directory paths from pipeline result
    train_dir = pipeline_result['train_dir']
    val_dir = pipeline_result['validation_dir']
    test_dir = pipeline_result['test_dir']
    
    print("\n" + "="*50)
    print("Starting Dataset Visualization")
    print("="*50)
    
    # Visualize the processed dataset
    visualize_dataset_distribution(
        dataset_dir=base_dir,
        result_dir=result_dir,
        data_type=data_type
    )
    
    # Train the models (VGG16, VGG19 and InceptionV3)
    model_scores = {}
    histories = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Use model-specific image size
        image_size = config['input_shape'][:2]
        
        # Create generators with model-specific preprocessing and size
        train_gen, val_gen, test_gen = setup_datagenerator(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            preprocess_fn=config['preprocess_fn'],
            batch_size=batch_size,
            image_size=image_size
        )
        
        # Train the model
        model, history = train_model(
            model_name=model_name,
            config=config,
            train_gen=train_gen,
            val_gen=val_gen,
            num_classes=num_classes,
            epochs=100
        )
        
        # Store history for plotting
        histories[model_name] = history
        
        # Plot the training history
        print("Plotting training and validation history")
        plot_training_history(
            history=history,
            model_name=model_name,
            save_dir=result_dir
        )
        
        # Evaluate model
        print("Evaluating model performance...")
        class_names = list(train_gen.class_indices.keys())  # get class labels
        test_accuracy, report = evaluate_model(
            model=model,
            test_generator=test_gen,
            model_name=model_name,
            class_names=class_names,
            save_dir=result_dir
        )
        
        # Save score for later comparison
        model_scores[model_name] = test_accuracy
        print("Model Report\n",report)
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Compare models
    plot_model_comparison(model_scores, result_dir)
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()