import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle

def evaluate_model(model, test_generator, model_name, class_names, save_dir):
    """Evaluate the model on test data and generate metrics and visualizations"""
    # Get predictions
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    
    # Calculate test accuracy
    test_accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"Classification Report for {model_name}:\n{report}")
     # ✅ Save classification report
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    # ROC Curve (for binary or multi-class classification)
    if len(class_names) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'roc_curve_{model_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC curve saved to {save_path}")
    else:
        # Multi-class classification
        # Binarize the output
        y_true_bin = tf.keras.utils.to_categorical(y_true, len(class_names))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'olive'])
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'roc_curve_{model_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC curve saved to {save_path}")
    
    return test_accuracy, report

# Compare the models
def plot_model_comparison(model_scores, save_dir):
    """Plot a comparison of model accuracies"""
    plt.figure(figsize=(10, 6))
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    bars = plt.bar(models, scores, color=['blue', 'green', 'red', 'purple', 'orange'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Model comparison plot saved to {save_path}")