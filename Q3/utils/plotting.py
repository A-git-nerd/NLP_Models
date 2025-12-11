import matplotlib.pyplot as plt
import os

def plot_training_history(histories, bleu_scores, model_names, save_dir='results'):
    """
    Plot training history for multiple seq2seq models
    histories: dict of {model_name: history_dict}
    bleu_scores: dict of {model_name: bleu_score}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Train and Val Loss
    ax1 = axes[0, 0]
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], label=f'{model_name} - Train', linestyle='-', marker='o', markersize=3)
            ax1.plot(epochs, history['val_loss'], label=f'{model_name} - Val', linestyle='--', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # Train Loss only (for clarity)
    ax2 = axes[0, 1]
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['train_loss']) + 1)
            ax2.plot(epochs, history['train_loss'], label=model_name, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True)
    
    # Validation Loss only (for clarity)
    ax3 = axes[1, 0]
    for model_name in model_names:
        if model_name in histories:
            history = histories[model_name]
            epochs = range(1, len(history['val_loss']) + 1)
            ax3.plot(epochs, history['val_loss'], label=model_name, marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Validation Loss Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True)
    
    # BLEU Scores Comparison
    ax4 = axes[1, 1]
    model_list = [name for name in model_names if name in bleu_scores]
    bleu_values = [bleu_scores[name] for name in model_list]
    colors = plt.cm.viridis(range(len(model_list)))
    bars = ax4.bar(range(len(model_list)), bleu_values, color=colors)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('BLEU Score')
    ax4.set_title('BLEU Score Comparison')
    ax4.set_xticks(range(len(model_list)))
    ax4.set_xticklabels(model_list, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, bleu_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to '{os.path.join(save_dir, 'training_metrics.png')}'")
    plt.close()
