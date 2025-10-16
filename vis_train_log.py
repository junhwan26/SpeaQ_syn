#!/usr/bin/env python3
"""
Visualize training logs from SpeaQ model training
Extracts train loss trends and validation metrics from log files
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import os

def parse_log_file(log_path):
    """Parse log file to extract training metrics"""
    
    # Patterns for different metrics
    train_loss_pattern = r'iter: (\d+).*?total_loss: ([\d.]+)'
    validation_pattern = r'copypaste: Task: (bbox|SG)\s*\n.*?copypaste: ([\d.,]+)'
    
    # Storage for metrics
    train_data = {
        'iterations': [],
        'total_loss': [],
        'loss_ce_subject': [],
        'loss_bbox_subject': [],
        'loss_giou_subject': [],
        'loss_ce_object': [],
        'loss_bbox_object': [],
        'loss_giou_object': [],
        'loss_relation': [],
        'loss_bbox_relation': [],
        'loss_giou_relation': [],
        'lr': []
    }
    
    val_data = {
        'iterations': [],
        'bbox_ap': [],
        'bbox_ap50': [],
        'bbox_ap75': [],
        'sg_mean_recall_20': [],
        'sg_mean_recall_50': [],
        'sg_mean_recall_100': [],
        'sg_recall_20': [],
        'sg_recall_50': [],
        'sg_recall_100': [],
        'recall_per_predicate': {}  # {'predicate_name': {'R@20': [], 'R@50': [], 'R@100': []}}
    }
    
    print(f"Parsing log file: {log_path}")
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract training losses
    train_matches = re.findall(train_loss_pattern, content)
    print(f"Found {len(train_matches)} training iterations")
    
    for match in train_matches:
        iteration = int(match[0])
        total_loss = float(match[1])
        train_data['iterations'].append(iteration)
        train_data['total_loss'].append(total_loss)
    
    # Extract detailed loss components (more complex pattern)
    detailed_loss_pattern = r'iter: (\d+).*?total_loss: ([\d.]+).*?loss_ce_subject: ([\d.]+).*?loss_bbox_subject: ([\d.]+).*?loss_giou_subject: ([\d.]+).*?loss_ce_object: ([\d.]+).*?loss_bbox_object: ([\d.]+).*?loss_giou_object: ([\d.]+).*?loss_relation: ([\d.]+).*?loss_bbox_relation: ([\d.]+).*?loss_giou_relation: ([\d.]+).*?lr: ([\d.e-]+)'
    
    detailed_matches = re.findall(detailed_loss_pattern, content, re.DOTALL)
    print(f"Found {len(detailed_matches)} detailed training iterations")
    
    for match in detailed_matches:
        iteration = int(match[0])
        total_loss = float(match[1])
        loss_ce_subject = float(match[2])
        loss_bbox_subject = float(match[3])
        loss_giou_subject = float(match[4])
        loss_ce_object = float(match[5])
        loss_bbox_object = float(match[6])
        loss_giou_object = float(match[7])
        loss_relation = float(match[8])
        loss_bbox_relation = float(match[9])
        loss_giou_relation = float(match[10])
        lr = float(match[11])
        
        train_data['loss_ce_subject'].append(loss_ce_subject)
        train_data['loss_bbox_subject'].append(loss_bbox_subject)
        train_data['loss_giou_subject'].append(loss_giou_subject)
        train_data['loss_ce_object'].append(loss_ce_object)
        train_data['loss_bbox_object'].append(loss_bbox_object)
        train_data['loss_giou_object'].append(loss_giou_object)
        train_data['loss_relation'].append(loss_relation)
        train_data['loss_bbox_relation'].append(loss_bbox_relation)
        train_data['loss_giou_relation'].append(loss_giou_relation)
        train_data['lr'].append(lr)
    
    # Extract validation metrics - improved pattern
    # Look for validation blocks with iteration numbers
    val_block_pattern = r'iter: (\d+).*?copypaste: Task: bbox\s*\n.*?copypaste: AP,AP50,AP75,APs,APm,APl\s*\n.*?copypaste: ([\d.,]+).*?copypaste: Task: SG\s*\n.*?copypaste: SGMeanRecall@20,SGMeanRecall@50,SGMeanRecall@100,SGRecall@20,SGRecall@50,SGRecall@100\s*\n.*?copypaste: ([\d.,]+)'
    
    val_matches = re.findall(val_block_pattern, content, re.DOTALL)
    print(f"Found {len(val_matches)} validation evaluations")
    
    for match in val_matches:
        iteration = int(match[0])
        bbox_metrics_str = match[1]
        sg_metrics_str = match[2]
        
        bbox_metrics = [float(x) for x in bbox_metrics_str.split(',')]
        sg_metrics = [float(x) for x in sg_metrics_str.split(',')]
        
        val_data['iterations'].append(iteration)
        
        # BBox metrics
        val_data['bbox_ap'].append(bbox_metrics[0])
        val_data['bbox_ap50'].append(bbox_metrics[1])
        val_data['bbox_ap75'].append(bbox_metrics[2])
        
        # Scene Graph metrics
        val_data['sg_mean_recall_20'].append(sg_metrics[0])
        val_data['sg_mean_recall_50'].append(sg_metrics[1])
        val_data['sg_mean_recall_100'].append(sg_metrics[2])
        val_data['sg_recall_20'].append(sg_metrics[3])
        val_data['sg_recall_50'].append(sg_metrics[4])
        val_data['sg_recall_100'].append(sg_metrics[5])
    
    # Extract recall per predicate
    # Pattern to find recall_per_class sections with iteration context
    recall_block_pattern = r'iter: (\d+).*?recall_per_class\s*\nR @ 20\s*\n(.*?)\nR @ 50\s*\n(.*?)\nR @ 100\s*\n(.*?)\n={50,}'
    
    recall_matches = re.findall(recall_block_pattern, content, re.DOTALL)
    print(f"Found {len(recall_matches)} recall_per_class evaluations")
    
    for match in recall_matches:
        iteration = int(match[0])
        recall_20_str = match[1].strip()
        recall_50_str = match[2].strip()
        recall_100_str = match[3].strip()
        
        # Parse recall@20 - improved pattern to capture all predicates including multi-word ones
        recall_20_items = re.findall(r'([\w\s]+?):\s+([\d.]+|nan);', recall_20_str)
        recall_50_items = re.findall(r'([\w\s]+?):\s+([\d.]+|nan);', recall_50_str)
        recall_100_items = re.findall(r'([\w\s]+?):\s+([\d.]+|nan);', recall_100_str)
        
        # Store recall per predicate
        for pred_name, value_str in recall_20_items:
            pred_name = pred_name.strip()
            if pred_name not in val_data['recall_per_predicate']:
                val_data['recall_per_predicate'][pred_name] = {
                    'iterations': [],
                    'R@20': [],
                    'R@50': [],
                    'R@100': []
                }
            
            # Only append if this iteration hasn't been added yet for this predicate
            if not val_data['recall_per_predicate'][pred_name]['iterations'] or \
               val_data['recall_per_predicate'][pred_name]['iterations'][-1] != iteration:
                val_data['recall_per_predicate'][pred_name]['iterations'].append(iteration)
                
                # Parse R@20
                try:
                    val_data['recall_per_predicate'][pred_name]['R@20'].append(float(value_str))
                except:
                    val_data['recall_per_predicate'][pred_name]['R@20'].append(0.0)
        
        # Parse R@50 and R@100
        for pred_name, value_str in recall_50_items:
            pred_name = pred_name.strip()
            if pred_name in val_data['recall_per_predicate']:
                try:
                    val_data['recall_per_predicate'][pred_name]['R@50'].append(float(value_str))
                except:
                    val_data['recall_per_predicate'][pred_name]['R@50'].append(0.0)
        
        for pred_name, value_str in recall_100_items:
            pred_name = pred_name.strip()
            if pred_name in val_data['recall_per_predicate']:
                try:
                    val_data['recall_per_predicate'][pred_name]['R@100'].append(float(value_str))
                except:
                    val_data['recall_per_predicate'][pred_name]['R@100'].append(0.0)
    
    return train_data, val_data

def plot_recall_per_predicate(recall_per_predicate, output_dir='./training_plots'):
    """Plot recall trends for each predicate"""
    
    if not recall_per_predicate:
        print("No recall_per_predicate data to plot")
        return
    
    # Filter predicates with valid data points (keep all predicates including zeros)
    valid_predicates = {}
    for pred_name, data in recall_per_predicate.items():
        if data['iterations'] and len(data['iterations']) > 0:
            valid_predicates[pred_name] = data
    
    if not valid_predicates:
        print("No valid predicate data to plot")
        return
    
    print(f"Plotting recall for all {len(valid_predicates)} predicates")
    
    # Sort predicates by final R@100 value (descending)
    sorted_predicates = sorted(valid_predicates.items(), 
                               key=lambda x: x[1]['R@100'][-1] if x[1]['R@100'] else 0, 
                               reverse=True)
    
    # Plot 1: ALL predicates recall trends (sorted by R@100)
    n_predicates = len(sorted_predicates)
    
    # Calculate grid size to fit all predicates
    n_cols = 5
    n_rows = (n_predicates + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    axes = axes.flatten()
    
    for idx, (pred_name, data) in enumerate(sorted_predicates):
        ax = axes[idx]
        ax.plot(data['iterations'], data['R@20'], 'o-', label='R@20', linewidth=2, markersize=4)
        ax.plot(data['iterations'], data['R@50'], 's-', label='R@50', linewidth=2, markersize=4)
        ax.plot(data['iterations'], data['R@100'], '^-', label='R@100', linewidth=2, markersize=4)
        ax.set_title(f'{pred_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel('Recall', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for idx in range(n_predicates, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_per_predicate_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved all {n_predicates} predicates to recall_per_predicate_all.png")
    
    # Removed individual per-predicate file saving
    
    # Plot 2: All predicates R@100 comparison (final values)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    pred_names = [pred_name for pred_name, _ in sorted_predicates]
    final_r100 = [data['R@100'][-1] if data['R@100'] else 0 for _, data in sorted_predicates]
    
    y_pos = np.arange(len(pred_names))
    bars = ax.barh(y_pos, final_r100, alpha=0.7)
    
    # Color bars by value
    colors = plt.cm.RdYlGn(np.array(final_r100) / max(final_r100) if max(final_r100) > 0 else np.array(final_r100))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pred_names, fontsize=8)
    ax.set_xlabel('Final Recall@100', fontsize=12, fontweight='bold')
    ax.set_title('Final Recall@100 per Predicate (Sorted)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_per_predicate_final_r100.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Removed important predicates plot generation
    
    print(f"Recall per predicate plots saved to: {output_dir}")

def plot_training_curves(train_data, val_data, output_dir='./training_plots'):
    """Create comprehensive training visualization plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    
    # 1. Training Overview Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total Loss
    if train_data['iterations'] and train_data['total_loss']:
        ax1.plot(train_data['iterations'], train_data['total_loss'], 'b-', linewidth=1, alpha=0.7)
        ax1.set_title('Total Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(train_data['total_loss']) > 100:
            window = min(100, len(train_data['total_loss']) // 10)
            moving_avg = np.convolve(train_data['total_loss'], np.ones(window)/window, mode='valid')
            ax1.plot(train_data['iterations'][window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            ax1.legend()
    
    # Validation Metrics - BBox
    if val_data['iterations'] and val_data['bbox_ap']:
        ax2.plot(val_data['iterations'], val_data['bbox_ap'], 'o-', label='AP', linewidth=2, markersize=4)
        ax2.plot(val_data['iterations'], val_data['bbox_ap50'], 's-', label='AP50', linewidth=2, markersize=4)
        ax2.plot(val_data['iterations'], val_data['bbox_ap75'], '^-', label='AP75', linewidth=2, markersize=4)
        ax2.set_title('Validation BBox Detection Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('AP Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Validation Metrics - Scene Graph Mean Recall
    if val_data['iterations'] and val_data['sg_mean_recall_20']:
        ax3.plot(val_data['iterations'], val_data['sg_mean_recall_20'], 'o-', label='Mean Recall@20', linewidth=2, markersize=4)
        ax3.plot(val_data['iterations'], val_data['sg_mean_recall_50'], 's-', label='Mean Recall@50', linewidth=2, markersize=4)
        ax3.plot(val_data['iterations'], val_data['sg_mean_recall_100'], '^-', label='Mean Recall@100', linewidth=2, markersize=4)
        ax3.set_title('Validation Scene Graph Mean Recall', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Mean Recall')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Set y-axis range to show more detail
        all_mean_recall_values = (val_data['sg_mean_recall_20'] + val_data['sg_mean_recall_50'] + 
                                 val_data['sg_mean_recall_100'])
        y_min = min(all_mean_recall_values) * 0.95
        y_max = max(all_mean_recall_values) * 1.05
        ax3.set_ylim(y_min, y_max)
    
    # Validation Metrics - Scene Graph Recall
    if val_data['iterations'] and val_data['sg_recall_20']:
        ax4.plot(val_data['iterations'], val_data['sg_recall_20'], 'o-', label='Recall@20', linewidth=2, markersize=4)
        ax4.plot(val_data['iterations'], val_data['sg_recall_50'], 's-', label='Recall@50', linewidth=2, markersize=4)
        ax4.plot(val_data['iterations'], val_data['sg_recall_100'], '^-', label='Recall@100', linewidth=2, markersize=4)
        ax4.set_title('Validation Scene Graph Recall', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Set y-axis range to show more detail
        all_recall_values = (val_data['sg_recall_20'] + val_data['sg_recall_50'] + 
                            val_data['sg_recall_100'])
        y_min = min(all_recall_values) * 0.95
        y_max = max(all_recall_values) * 1.05
        ax4.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Loss Components Plot
    if train_data['loss_ce_subject']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        loss_components = [
            ('loss_ce_subject', 'Subject Classification Loss'),
            ('loss_bbox_subject', 'Subject BBox Loss'),
            ('loss_ce_object', 'Object Classification Loss'),
            ('loss_bbox_object', 'Object BBox Loss')
        ]
        
        for i, (loss_key, title) in enumerate(loss_components):
            if i < len(axes) and train_data[loss_key]:
                axes[i].plot(train_data['iterations'][:len(train_data[loss_key])], 
                           train_data[loss_key], 'b-', linewidth=1, alpha=0.7)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_loss_components.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Relation Loss Components
    if train_data['loss_relation']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        relation_components = [
            ('loss_relation', 'Relation Classification Loss'),
            ('loss_bbox_relation', 'Relation BBox Loss'),
            ('loss_giou_relation', 'Relation GIoU Loss')
        ]
        
        for i, (loss_key, title) in enumerate(relation_components):
            if train_data[loss_key]:
                axes[i].plot(train_data['iterations'][:len(train_data[loss_key])], 
                           train_data[loss_key], 'r-', linewidth=1, alpha=0.7)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relation_loss_components.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Validation Metrics Detailed - Separate Mean Recall and Recall
    if val_data['iterations'] and val_data['sg_recall_20']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scene Graph Mean Recall
        ax1.plot(val_data['iterations'], val_data['sg_mean_recall_20'], 'o-', label='Mean Recall@20', linewidth=2, markersize=4)
        ax1.plot(val_data['iterations'], val_data['sg_mean_recall_50'], 's-', label='Mean Recall@50', linewidth=2, markersize=4)
        ax1.plot(val_data['iterations'], val_data['sg_mean_recall_100'], '^-', label='Mean Recall@100', linewidth=2, markersize=4)
        ax1.set_title('Validation Scene Graph Mean Recall', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Recall')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Set y-axis range to show more detail for Mean Recall
        all_mean_recall_values = (val_data['sg_mean_recall_20'] + val_data['sg_mean_recall_50'] + 
                                 val_data['sg_mean_recall_100'])
        y_min = min(all_mean_recall_values) * 0.95
        y_max = max(all_mean_recall_values) * 1.05
        ax1.set_ylim(y_min, y_max)
        
        # Scene Graph Recall
        ax2.plot(val_data['iterations'], val_data['sg_recall_20'], 'o-', label='Recall@20', linewidth=2, markersize=4)
        ax2.plot(val_data['iterations'], val_data['sg_recall_50'], 's-', label='Recall@50', linewidth=2, markersize=4)
        ax2.plot(val_data['iterations'], val_data['sg_recall_100'], '^-', label='Recall@100', linewidth=2, markersize=4)
        ax2.set_title('Validation Scene Graph Recall', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Set y-axis range to show more detail for Recall
        all_recall_values = (val_data['sg_recall_20'] + val_data['sg_recall_50'] + 
                            val_data['sg_recall_100'])
        y_min = min(all_recall_values) * 0.95
        y_max = max(all_recall_values) * 1.05
        ax2.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_metrics_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Combined Metrics Plot
    if val_data['iterations'] and val_data['sg_recall_20']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Combined metrics
        ax.plot(val_data['iterations'], val_data['bbox_ap'], 'o-', label='BBox AP', linewidth=2, markersize=6)
        ax.plot(val_data['iterations'], val_data['sg_mean_recall_50'], 's-', label='SG Mean Recall@50', linewidth=2, markersize=6)
        ax.plot(val_data['iterations'], val_data['sg_recall_50'], '^-', label='SG Recall@50', linewidth=2, markersize=6)
        ax.set_title('Combined Validation Metrics', fontsize=16, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        # Keep combined plot fixed to [0,1] for absolute comparison
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_validation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Recall Per Predicate Plots
    if val_data['recall_per_predicate']:
        plot_recall_per_predicate(val_data['recall_per_predicate'], output_dir)
    
    print(f"Training plots saved to: {output_dir}")

def print_training_summary(train_data, val_data):
    """Print a summary of training progress"""
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if train_data['iterations']:
        print(f"Total Training Iterations: {len(train_data['iterations'])}")
        print(f"Final Iteration: {max(train_data['iterations'])}")
        
        if train_data['total_loss']:
            print(f"Initial Total Loss: {train_data['total_loss'][0]:.4f}")
            print(f"Final Total Loss: {train_data['total_loss'][-1]:.4f}")
            print(f"Loss Reduction: {train_data['total_loss'][0] - train_data['total_loss'][-1]:.4f}")
        
        if train_data['lr']:
            print(f"Initial Learning Rate: {train_data['lr'][0]:.2e}")
            print(f"Final Learning Rate: {train_data['lr'][-1]:.2e}")
    
    if val_data['iterations']:
        print(f"\nValidation Evaluations: {len(val_data['iterations'])}")
        
        if val_data['bbox_ap']:
            print(f"Best BBox AP: {max(val_data['bbox_ap']):.4f}")
            print(f"Final BBox AP: {val_data['bbox_ap'][-1]:.4f}")
        
        if val_data['sg_mean_recall_50']:
            print(f"Best SG Mean Recall@50: {max(val_data['sg_mean_recall_50']):.4f}")
            print(f"Final SG Mean Recall@50: {val_data['sg_mean_recall_50'][-1]:.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize SpeaQ training logs')
    parser.add_argument('--log-path', required=True, help='Path to the log file')
    parser.add_argument('--output-dir', default='./training_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file not found: {args.log_path}")
        return
    
    # Parse the log file
    train_data, val_data = parse_log_file(args.log_path)
    
    # Print summary
    print_training_summary(train_data, val_data)
    
    # Create plots
    plot_training_curves(train_data, val_data, args.output_dir)
    
    print(f"\nVisualization complete! Check the plots in: {args.output_dir}")

if __name__ == "__main__":
    main()
