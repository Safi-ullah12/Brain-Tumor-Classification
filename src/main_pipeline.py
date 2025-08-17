from Dataset_pipeline import create_dataset_pipeline
from EDA_origanal_preprocess import visualize_dataset_distribution
from model_utils import setup_datagenerator

def main():
    # Define  parameters directly in the script
    source_dir = "path/to/your/source_dataset"  # Update with your actual path
    base_dir = "path/to/your/processed_dataset"  # Update with your actual path
    result_dir = "path/to/your/results"  # Update with your actual path
    image_size = (224, 224)
    val_ratio = 0.10
    batch_size = 32  # Added missing batch_size parameter
    data_type = "preprocessed"  # or "original"
    
    print("="*50)
    print("Starting Dataset Processing Pipeline")
    print("="*50)
    
    # Process the dataset
    pipeline_result = create_dataset_pipeline(
        source_dir=source_dir,
        base_dir=base_dir,
        image_size=image_size,
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
    
    print("\n" + "="*50)
    print("Setting up Data Generators")
    print("="*50)
    
    # Create the generators with all required parameters
    train_gen, val_gen, test_gen = setup_datagenerator(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        preprocess_fn=preprocess_input,  # Added missing preprocess function
        batch_size=batch_size,
        image_size=image_size
    )
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(f"Processed dataset saved to: {base_dir}")
    print(f"Visualization results saved to: {result_dir}")
    print("="*50)

if __name__ == "__main__":
    main()