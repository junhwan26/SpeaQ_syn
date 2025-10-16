#!/usr/bin/env python3
"""
Simple Predicate Distribution Analysis Tool for SpeaQ

This script analyzes predicate distributions without complex visualizations.
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse

def load_predicate_data(mapping_dict_path, h5_path):
    """Load predicate data from H5 file"""
    print(f"Loading mapping dictionary from: {mapping_dict_path}")
    with open(mapping_dict_path, 'r') as f:
        mapping_dict = json.load(f)
    
    predicate_to_idx = mapping_dict['predicate_to_idx']
    idx_to_predicates = sorted(predicate_to_idx.keys(), key=lambda k: predicate_to_idx[k])
    
    print(f"Loading annotations from: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        predicates = f['predicates'][:]
        
    print(f"Found {len(predicates)} relationships")
    
    # Count predicate occurrences
    predicate_counts = Counter(predicates.flatten())
    
    return predicate_counts, idx_to_predicates

def load_config_paths(config_file):
    """Load dataset paths from config YAML file"""
    import yaml
    
    print(f"Loading config from: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets_config = config.get('DATASETS', {})
    dataset_type = datasets_config.get('TYPE', '')
    
    if dataset_type == "MULTI_DATASET":
        visual_genome = datasets_config.get('VISUAL_GENOME', {})
        visual_genome_synthetic = datasets_config.get('VISUAL_GENOME_SYNTHETIC', {})
        
        real_mapping = visual_genome.get('MAPPING_DICTIONARY', '')
        real_h5 = visual_genome.get('VG_ATTRIBUTE_H5', '')
        synthetic_mapping = visual_genome_synthetic.get('MAPPING_DICTIONARY', '')
        synthetic_h5 = visual_genome_synthetic.get('VG_ATTRIBUTE_H5', '')
        print("✓ Multi-dataset config loaded")
        
    elif dataset_type == "SYNTHETIC GENOME":
        synthetic_genome = datasets_config.get('SYNTHETIC_GENOME', {})
        
        real_mapping = None
        real_h5 = None
        synthetic_mapping = synthetic_genome.get('MAPPING_DICTIONARY', '')
        synthetic_h5 = synthetic_genome.get('VG_ATTRIBUTE_H5', '')
        print("✓ Synthetic-only config loaded")
        
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return real_mapping, real_h5, synthetic_mapping, synthetic_h5

def create_simple_plots(real_counts, real_idx_to_pred, synth_counts, synth_idx_to_pred, output_dir):
    """Create beautiful predicate distribution plots using matplotlib"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # 1. Top predicates comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Real dataset
    top_real = real_counts.most_common(15)
    real_preds = []
    real_values = []
    for idx, count in top_real:
        if idx < len(real_idx_to_pred):
            real_preds.append(real_idx_to_pred[idx])
        else:
            real_preds.append(f"pred_{idx}")
        real_values.append(count)
    
    # Create horizontal bar plot with matplotlib
    bars1 = ax1.barh(range(len(real_preds)), real_values, color=colors[0], alpha=0.8)
    ax1.set_yticks(range(len(real_preds)))
    ax1.set_yticklabels(real_preds, fontsize=9)
    ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicates', fontsize=12, fontweight='bold')
    ax1.set_title('Top 15 Predicates - Real Dataset', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Synthetic dataset
    top_synth = synth_counts.most_common(15)
    synth_preds = []
    synth_values = []
    for idx, count in top_synth:
        if idx < len(synth_idx_to_pred):
            synth_preds.append(synth_idx_to_pred[idx])
        else:
            synth_preds.append(f"pred_{idx}")
        synth_values.append(count)
    
    # Create horizontal bar plot with matplotlib
    bars2 = ax2.barh(range(len(synth_preds)), synth_values, color=colors[1], alpha=0.8)
    ax2.set_yticks(range(len(synth_preds)))
    ax2.set_yticklabels(synth_preds, fontsize=9)
    ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicates', fontsize=12, fontweight='bold')
    ax2.set_title('Top 15 Predicates - Synthetic Dataset', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution comparison plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get all common predicates
    common_preds = set(real_counts.keys()) & set(synth_counts.keys())
    pred_data = []
    
    for idx in sorted(common_preds):
        if idx < len(real_idx_to_pred):
            pred_name = real_idx_to_pred[idx]
        else:
            pred_name = f"pred_{idx}"
        
        pred_data.append({
            'Predicate': pred_name,
            'Real': real_counts[idx],
            'Synthetic': synth_counts[idx]
        })
    
    # Create grouped bar plot with matplotlib
    x_pos = np.arange(len(pred_data))
    width = 0.35
    
    real_vals = [item['Real'] for item in pred_data]
    synth_vals = [item['Synthetic'] for item in pred_data]
    pred_names = [item['Predicate'] for item in pred_data]
    
    bars1 = ax.bar(x_pos - width/2, real_vals, width, label='Real Dataset', 
                   color=colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, synth_vals, width, label='Synthetic Dataset', 
                   color=colors[1], alpha=0.8)
    
    ax.set_title('Predicate Distribution Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicates', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pred_names, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Dataset', title_fontsize=12, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicate_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Log scale comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create log scale plot with matplotlib
    bars1 = ax.bar(x_pos - width/2, real_vals, width, label='Real Dataset', 
                   color=colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, synth_vals, width, label='Synthetic Dataset', 
                   color=colors[1], alpha=0.8)
    
    ax.set_yscale('log')
    ax.set_title('Predicate Distribution Comparison (Log Scale)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicates', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pred_names, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Dataset', title_fontsize=12, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicate_distribution_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved:")
    print(f"  - {output_dir}/predicate_comparison.png")
    print(f"  - {output_dir}/predicate_distribution_comparison.png") 
    print(f"  - {output_dir}/predicate_distribution_log.png")

def analyze_overlap(real_counts, real_idx_to_pred, synth_counts, synth_idx_to_pred):
    """Analyze predicate overlap between datasets"""
    real_pred_indices = set(real_counts.keys())
    synth_pred_indices = set(synth_counts.keys())
    
    common_predicates = real_pred_indices & synth_pred_indices
    real_only = real_pred_indices - synth_pred_indices
    synth_only = synth_pred_indices - real_pred_indices
    
    print(f"\n=== Predicate Overlap Analysis ===")
    print(f"Real dataset unique predicates: {len(real_pred_indices)}")
    print(f"Synthetic dataset unique predicates: {len(synth_pred_indices)}")
    print(f"Common predicates: {len(common_predicates)}")
    print(f"Real-only predicates: {len(real_only)}")
    print(f"Synthetic-only predicates: {len(synth_only)}")
    
    # Find predicates with significant frequency differences
    significant_diff = []
    for pred_idx in common_predicates:
        real_count = real_counts[pred_idx]
        synth_count = synth_counts[pred_idx]
        
        if real_count > 0 and synth_count > 0:
            rel_diff = abs(real_count - synth_count) / max(real_count, synth_count)
            if rel_diff > 0.5:  # More than 50% difference
                if pred_idx < len(real_idx_to_pred):
                    pred_name = real_idx_to_pred[pred_idx]
                else:
                    pred_name = f"pred_{pred_idx}"
                significant_diff.append((pred_name, real_count, synth_count, rel_diff))
    
    if significant_diff:
        print(f"\n=== Predicates with Significant Frequency Differences (>50%) ===")
        significant_diff.sort(key=lambda x: x[3], reverse=True)
        for pred_name, real_count, synth_count, rel_diff in significant_diff[:10]:
            print(f"{pred_name}: Real={real_count}, Synthetic={synth_count}, Diff={rel_diff:.1%}")

def main():
    parser = argparse.ArgumentParser(description='Simple predicate distribution analysis')
    parser.add_argument('--config-file', type=str,
                       default='/home/junhwanheo/SpeaQ/configs/speaq_multi_dataset.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output-dir', type=str, default='predicate_analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("=== Simple Predicate Distribution Analysis ===")
    print(f"Config file: {args.config_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Load config
    try:
        real_mapping, real_h5, synthetic_mapping, synthetic_h5 = load_config_paths(args.config_file)
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return
    
    # Load real dataset
    real_data = None
    if real_mapping and real_h5:
        print(f"\nLoading real dataset...")
        try:
            real_counts, real_idx_to_pred = load_predicate_data(real_mapping, real_h5)
            real_data = (real_counts, real_idx_to_pred)
            print(f"✓ Real dataset loaded: {len(real_counts)} unique predicates")
        except Exception as e:
            print(f"✗ Error loading real dataset: {e}")
    
    # Load synthetic dataset
    synth_data = None
    if synthetic_mapping and synthetic_h5:
        print(f"\nLoading synthetic dataset...")
        try:
            synth_counts, synth_idx_to_pred = load_predicate_data(synthetic_mapping, synthetic_h5)
            synth_data = (synth_counts, synth_idx_to_pred)
            print(f"✓ Synthetic dataset loaded: {len(synth_counts)} unique predicates")
        except Exception as e:
            print(f"✗ Error loading synthetic dataset: {e}")
    
    # Analyze if both datasets are available
    if real_data and synth_data:
        real_counts, real_idx_to_pred = real_data
        synth_counts, synth_idx_to_pred = synth_data
        
        # Analyze overlap
        analyze_overlap(real_counts, real_idx_to_pred, synth_counts, synth_idx_to_pred)
        
        # Create plots
        print(f"\nCreating visualizations...")
        try:
            create_simple_plots(real_counts, real_idx_to_pred, synth_counts, synth_idx_to_pred, args.output_dir)
        except Exception as e:
            print(f"✗ Error creating plots: {e}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
