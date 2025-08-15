import os
import shutil
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def create_dataset_pipeline(source_dir, base_dir, image_size=(224, 224), val_ratio=0.20):
    """
    Augments train & test images separately.
    Splits only the train set into train and validation sets.
    No duplicate augmented dataset stored — only final train, val, and test folders.
    Shows progress with tqdm.
    """
    print("Starting Data Engineering Pipeline...")

    # Final directories
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Remove old output
    if os.path.exists(base_dir):
        print(f"Removing old base directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def augment_images(src_dir):
        """Resize + rotate + flip images and return augmented images in memory."""
        class_labels = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
        all_files = []
        all_labels = []
        total_original = 0
        total_augmented = 0

        print(f"Processing directory: {src_dir}")
        for class_name in class_labels:
            src_class_dir = os.path.join(src_dir, class_name)
            files = os.listdir(src_class_dir)
            total_original += len(files)

            for filename in tqdm(files, desc=f"Augmenting {class_name}", unit="img"):
                img_path = os.path.join(src_class_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                resized = cv2.resize(img, (image_size[1], image_size[0]))
                rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
                flipped = cv2.flip(resized, 0)

                base_name, ext = os.path.splitext(filename)
                aug_set = [
                    (f"{base_name}_orig{ext}", resized),
                    (f"{base_name}_rot{ext}", rotated),
                    (f"{base_name}_flip{ext}", flipped)
                ]

                for new_name, new_img in aug_set:
                    all_files.append((class_name, new_name, new_img))
                    all_labels.append(class_name)
                    total_augmented += 1

        print(f"Original images: {total_original}")
        print(f"Augmented images (including original resized): {total_augmented}")
        return all_files, all_labels, class_labels

    # Augment TRAIN
    print("\n=== Augmenting TRAIN set ===")
    train_files_aug, train_labels_aug, class_labels = augment_images(os.path.join(source_dir, 'train'))

    # Split augmented train into train/validation
    print("\n=== Splitting TRAIN set into train/validation ===")
    train_files_list, val_files_list, train_labels_list, val_labels_list = train_test_split(
        train_files_aug, train_labels_aug,
        test_size=val_ratio,
        random_state=42,
        stratify=train_labels_aug
    )

    def save_images(file_tuples, dest_dir):
        for cls, fname, img in tqdm(file_tuples, desc=f"Saving to {os.path.basename(dest_dir)}", unit="img"):
            cls_dir = os.path.join(dest_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cls_dir, fname), img)

    # Save final TRAIN and VALIDATION sets
    save_images(train_files_list, train_dir)
    save_images(val_files_list, validation_dir)

    # Augment TEST
    print("\n=== Augmenting TEST set ===")
    test_files_aug, _, _ = augment_images(os.path.join(source_dir, 'test'))
    save_images(test_files_aug, test_dir)

    # Final stats
    print("\nPipeline finished successfully.")
    print(f"Final TRAIN images: {len(train_files_list)}")
    print(f"Final VALIDATION images: {len(val_files_list)}")
    print(f"Final TEST images: {len(test_files_aug)}")

    return {
        'train_dir': train_dir,
        'validation_dir': validation_dir,
        'test_dir': test_dir,
        'class_labels': class_labels
    }
