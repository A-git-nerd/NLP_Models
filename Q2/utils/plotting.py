import matplotlib.pyplot as plt
import os

def plot_training_history(histories, model_names, save_dir='results'):
    """
    Plot training history for multiple models
    histories: dict of {model_name: history_dict}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(15, 10))
    
    # Train and Val Loss
    plt.subplot(2, 3, 1)
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['train_loss']) + 1)
            plt.plot(epochs, history['train_loss'], label=f'{model_name} - Train', linestyle='-')
            plt.plot(epochs, history['val_loss'], label=f'{model_name} - Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    # Accuracy
    plt.subplot(2, 3, 2)
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['train_acc']) + 1)
            plt.plot(epochs, history['train_acc'], label=f'{model_name} - Train', linestyle='-')
            plt.plot(epochs, history['val_acc'], label=f'{model_name} - Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    # Precision
    plt.subplot(2, 3, 3)
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['val_precision']) + 1)
            plt.plot(epochs, history['val_precision'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Validation Precision')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    # Recall
    plt.subplot(2, 3, 4)
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['val_recall']) + 1)
            plt.plot(epochs, history['val_recall'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Validation Recall')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    # F1 Score
    plt.subplot(2, 3, 5)
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['val_f1']) + 1)
            plt.plot(epochs, history['val_f1'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    # Comparison of final metrics
    plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = range(len(metrics))
    width = 0.2
    
    for i, model_name in enumerate(model_names):
        if model_name in histories:
            history = histories[model_name]
            values = [
                history['val_acc'][-1],
                history['val_precision'][-1],
                history['val_recall'][-1],
                history['val_f1'][-1]
            ]
            plt.bar([pos + width * i for pos in x], values, width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Final Validation Metrics Comparison')
    plt.xticks([pos + width * (len(model_names) - 1) / 2 for pos in x], metrics)
    plt.legend(fontsize=8)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to '{os.path.join(save_dir, 'training_metrics.png')}'")
    plt.close()
