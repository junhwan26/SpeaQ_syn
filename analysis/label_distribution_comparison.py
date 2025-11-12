#!/usr/bin/env python3
"""
Label Distribution Comparison Tool for SpeaQ

This script compares label distributions between Real data (Visual Genome) 
and Synthetic data to analyze differences in object classes, predicates, and attributes.
"""

import os
import sys
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
import yaml
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LabelDistributionAnalyzer:
    """Analyzer for comparing label distributions between real and synthetic data"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self._load_config()
        self.real_data = None
        self.synthetic_data = None
        
    def _load_config(self):
        """Load configuration from YAML file"""
        print(f"Loading config from: {self.config_file}")
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_real_data(self):
        """Load real data (Visual Genome) statistics"""
        print("\n=== Loading Real Data (Visual Genome) ===")
        
        vg_config = self.config['DATASETS']['VISUAL_GENOME']
        mapping_dict_path = vg_config['MAPPING_DICTIONARY']
        h5_path = vg_config['VG_ATTRIBUTE_H5']
        
        print(f"Mapping dictionary: {mapping_dict_path}")
        print(f"H5 file: {h5_path}")
        
        # Load mapping dictionary
        with open(mapping_dict_path, 'r') as f:
            mapping_dict = json.load(f)
        
        # Load H5 data
        with h5py.File(h5_path, 'r') as f:
            # Get training split data
            split_mask = f['split'][:] == 0  # 0 = train, 1 = val, 2 = test
            split_mask &= f['img_to_first_box'][:] >= 0
            
            # Load labels and predicates
            all_labels = f['labels'][:, 0]
            all_predicates = f['predicates'][:, 0]
            
            # Get training data indices
            first_box_idx = f['img_to_first_box'][split_mask]
            last_box_idx = f['img_to_last_box'][split_mask]
            first_rel_idx = f['img_to_first_rel'][split_mask]
            last_rel_idx = f['img_to_last_rel'][split_mask]
            
            # Extract training data
            train_labels = []
            train_predicates = []
            
            for i in range(len(first_box_idx)):
                # Extract object labels
                if first_box_idx[i] >= 0:
                    labels = all_labels[first_box_idx[i]:last_box_idx[i]+1]
                    train_labels.extend(labels)
                
                # Extract predicates
                if first_rel_idx[i] >= 0:
                    predicates = all_predicates[first_rel_idx[i]:last_rel_idx[i]+1]
                    train_predicates.extend(predicates)
        
        # Convert to 0-based indexing
        train_labels = np.array(train_labels) - 1
        train_predicates = np.array(train_predicates) - 1
        
        # Create label mappings
        idx_to_classes = sorted(mapping_dict['label_to_idx'], 
                               key=lambda k: mapping_dict['label_to_idx'][k])
        idx_to_predicates = sorted(mapping_dict['predicate_to_idx'], 
                                 key=lambda k: mapping_dict['predicate_to_idx'][k])
        
        self.real_data = {
            'labels': train_labels,
            'predicates': train_predicates,
            'idx_to_classes': idx_to_classes,
            'idx_to_predicates': idx_to_predicates,
            'label_counts': Counter(train_labels),
            'predicate_counts': Counter(train_predicates)
        }
        
        print(f"✓ Real data loaded:")
        print(f"  - Total objects: {len(train_labels)}")
        print(f"  - Total relations: {len(train_predicates)}")
        print(f"  - Unique object classes: {len(set(train_labels))}")
        print(f"  - Unique predicates: {len(set(train_predicates))}")
        
        return self.real_data
    
    def load_synthetic_data(self):
        """Load synthetic data statistics"""
        print("\n=== Loading Synthetic Data ===")
        
        vg_syn_config = self.config['DATASETS']['VISUAL_GENOME_SYNTHETIC']
        mapping_dict_path = vg_syn_config['MAPPING_DICTIONARY']
        h5_path = vg_syn_config['VG_ATTRIBUTE_H5']
        
        print(f"Mapping dictionary: {mapping_dict_path}")
        print(f"H5 file: {h5_path}")
        
        # Load mapping dictionary
        with open(mapping_dict_path, 'r') as f:
            mapping_dict = json.load(f)
        
        # Load H5 data
        with h5py.File(h5_path, 'r') as f:
            # Get training split data
            split_mask = f['split'][:] == 0  # 0 = train, 1 = val, 2 = test
            split_mask &= f['img_to_first_box'][:] >= 0
            
            # Load labels and predicates
            all_labels = f['labels'][:, 0]
            all_predicates = f['predicates'][:, 0]
            
            # Get training data indices
            first_box_idx = f['img_to_first_box'][split_mask]
            last_box_idx = f['img_to_last_box'][split_mask]
            first_rel_idx = f['img_to_first_rel'][split_mask]
            last_rel_idx = f['img_to_last_rel'][split_mask]
            
            # Extract training data
            train_labels = []
            train_predicates = []
            
            for i in range(len(first_box_idx)):
                # Extract object labels
                if first_box_idx[i] >= 0:
                    labels = all_labels[first_box_idx[i]:last_box_idx[i]+1]
                    train_labels.extend(labels)
                
                # Extract predicates
                if first_rel_idx[i] >= 0:
                    predicates = all_predicates[first_rel_idx[i]:last_rel_idx[i]+1]
                    train_predicates.extend(predicates)
        
        # Convert to 0-based indexing
        train_labels = np.array(train_labels) - 1
        train_predicates = np.array(train_predicates) - 1
        
        # Create label mappings
        idx_to_classes = sorted(mapping_dict['label_to_idx'], 
                               key=lambda k: mapping_dict['label_to_idx'][k])
        idx_to_predicates = sorted(mapping_dict['predicate_to_idx'], 
                                 key=lambda k: mapping_dict['predicate_to_idx'][k])
        
        self.synthetic_data = {
            'labels': train_labels,
            'predicates': train_predicates,
            'idx_to_classes': idx_to_classes,
            'idx_to_predicates': idx_to_predicates,
            'label_counts': Counter(train_labels),
            'predicate_counts': Counter(train_predicates)
        }
        
        print(f"✓ Synthetic data loaded:")
        print(f"  - Total objects: {len(train_labels)}")
        print(f"  - Total relations: {len(train_predicates)}")
        print(f"  - Unique object classes: {len(set(train_labels))}")
        print(f"  - Unique predicates: {len(set(train_predicates))}")
        
        return self.synthetic_data
    
    def analyze_object_class_distribution(self):
        """Analyze object class distribution differences"""
        print("\n=== Object Class Distribution Analysis ===")
        
        real_labels = self.real_data['labels']
        synth_labels = self.synthetic_data['labels']
        
        # Get all unique classes
        all_classes = set(real_labels) | set(synth_labels)
        real_classes = set(real_labels)
        synth_classes = set(synth_labels)
        
        print(f"Real data unique classes: {len(real_classes)}")
        print(f"Synthetic data unique classes: {len(synth_classes)}")
        print(f"Common classes: {len(real_classes & synth_classes)}")
        print(f"Real-only classes: {len(real_classes - synth_classes)}")
        print(f"Synthetic-only classes: {len(synth_classes - real_classes)}")
        
        # Calculate class frequencies
        real_freq = Counter(real_labels)
        synth_freq = Counter(synth_labels)
        
        # Normalize frequencies
        real_total = len(real_labels)
        synth_total = len(synth_labels)
        
        real_normalized = {cls: count/real_total for cls, count in real_freq.items()}
        synth_normalized = {cls: count/synth_total for cls, count in synth_freq.items()}
        
        # Find classes with significant frequency differences
        significant_diffs = []
        for cls in all_classes:
            real_count = real_freq.get(cls, 0)
            synth_count = synth_freq.get(cls, 0)
            
            if real_count > 0 and synth_count > 0:
                real_pct = real_count / real_total
                synth_pct = synth_count / synth_total
                diff_pct = abs(real_pct - synth_pct)
                
                if diff_pct > 0.01:  # More than 1% difference
                    cls_name = self.real_data['idx_to_classes'][cls] if cls < len(self.real_data['idx_to_classes']) else f"class_{cls}"
                    significant_diffs.append((cls_name, real_pct, synth_pct, diff_pct))
        
        significant_diffs.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\nClasses with significant frequency differences (>1%):")
        for cls_name, real_pct, synth_pct, diff_pct in significant_diffs[:20]:
            print(f"  {cls_name}: Real={real_pct:.3f}, Synthetic={synth_pct:.3f}, Diff={diff_pct:.3f}")
        
        return {
            'all_classes': all_classes,
            'real_classes': real_classes,
            'synth_classes': synth_classes,
            'real_freq': real_freq,
            'synth_freq': synth_freq,
            'significant_diffs': significant_diffs
        }
    
    def analyze_predicate_distribution(self):
        """Analyze predicate distribution differences"""
        print("\n=== Predicate Distribution Analysis ===")
        
        real_predicates = self.real_data['predicates']
        synth_predicates = self.synthetic_data['predicates']
        
        # Get all unique predicates
        all_predicates = set(real_predicates) | set(synth_predicates)
        real_pred_set = set(real_predicates)
        synth_pred_set = set(synth_predicates)
        
        print(f"Real data unique predicates: {len(real_pred_set)}")
        print(f"Synthetic data unique predicates: {len(synth_pred_set)}")
        print(f"Common predicates: {len(real_pred_set & synth_pred_set)}")
        print(f"Real-only predicates: {len(real_pred_set - synth_pred_set)}")
        print(f"Synthetic-only predicates: {len(synth_pred_set - real_pred_set)}")
        
        # Calculate predicate frequencies
        real_freq = Counter(real_predicates)
        synth_freq = Counter(synth_predicates)
        
        # Normalize frequencies
        real_total = len(real_predicates)
        synth_total = len(synth_predicates)
        
        # Find predicates with significant frequency differences
        significant_diffs = []
        for pred in all_predicates:
            real_count = real_freq.get(pred, 0)
            synth_count = synth_freq.get(pred, 0)
            
            if real_count > 0 and synth_count > 0:
                real_pct = real_count / real_total
                synth_pct = synth_count / synth_total
                diff_pct = abs(real_pct - synth_pct)
                
                if diff_pct > 0.005:  # More than 0.5% difference
                    pred_name = self.real_data['idx_to_predicates'][pred] if pred < len(self.real_data['idx_to_predicates']) else f"pred_{pred}"
                    significant_diffs.append((pred_name, real_pct, synth_pct, diff_pct))
        
        significant_diffs.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\nPredicates with significant frequency differences (>0.5%):")
        for pred_name, real_pct, synth_pct, diff_pct in significant_diffs[:20]:
            print(f"  {pred_name}: Real={real_pct:.4f}, Synthetic={synth_pct:.4f}, Diff={diff_pct:.4f}")
        
        return {
            'all_predicates': all_predicates,
            'real_predicates': real_pred_set,
            'synth_predicates': synth_pred_set,
            'real_freq': real_freq,
            'synth_freq': synth_freq,
            'significant_diffs': significant_diffs
        }
    
    def create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        print(f"\n=== Creating Visualizations ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Object Class Distribution Comparison
        self._plot_object_class_distribution(output_dir)
        
        # 2. Predicate Distribution Comparison
        self._plot_predicate_distribution(output_dir)
        
        # 3. Top Classes Comparison
        self._plot_top_classes_comparison(output_dir)
        
        # 4. Top Predicates Comparison
        self._plot_top_predicates_comparison(output_dir)
        
        # 5. Distribution Statistics
        self._plot_distribution_statistics(output_dir)
        
        print(f"✓ Visualizations saved to: {output_dir}")
    
    def _plot_object_class_distribution(self, output_dir):
        """Plot object class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Real data
        real_counts = self.real_data['label_counts']
        top_real = real_counts.most_common(20)
        real_classes = [self.real_data['idx_to_classes'][idx] if idx < len(self.real_data['idx_to_classes']) else f"class_{idx}" 
                       for idx, _ in top_real]
        real_values = [count for _, count in top_real]
        
        bars1 = ax1.barh(range(len(real_classes)), real_values, color='skyblue', alpha=0.8)
        ax1.set_yticks(range(len(real_classes)))
        ax1.set_yticklabels(real_classes, fontsize=10)
        ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Object Classes - Real Data', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Synthetic data
        synth_counts = self.synthetic_data['label_counts']
        top_synth = synth_counts.most_common(20)
        synth_classes = [self.synthetic_data['idx_to_classes'][idx] if idx < len(self.synthetic_data['idx_to_classes']) else f"class_{idx}" 
                        for idx, _ in top_synth]
        synth_values = [count for _, count in top_synth]
        
        bars2 = ax2.barh(range(len(synth_classes)), synth_values, color='lightcoral', alpha=0.8)
        ax2.set_yticks(range(len(synth_classes)))
        ax2.set_yticklabels(synth_classes, fontsize=10)
        ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Top 20 Object Classes - Synthetic Data', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'object_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predicate_distribution(self, output_dir):
        """Plot predicate distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Real data
        real_counts = self.real_data['predicate_counts']
        top_real = real_counts.most_common(20)
        real_preds = [self.real_data['idx_to_predicates'][idx] if idx < len(self.real_data['idx_to_predicates']) else f"pred_{idx}" 
                     for idx, _ in top_real]
        real_values = [count for _, count in top_real]
        
        bars1 = ax1.barh(range(len(real_preds)), real_values, color='lightgreen', alpha=0.8)
        ax1.set_yticks(range(len(real_preds)))
        ax1.set_yticklabels(real_preds, fontsize=10)
        ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Predicates - Real Data', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Synthetic data
        synth_counts = self.synthetic_data['predicate_counts']
        top_synth = synth_counts.most_common(20)
        synth_preds = [self.synthetic_data['idx_to_predicates'][idx] if idx < len(self.synthetic_data['idx_to_predicates']) else f"pred_{idx}" 
                      for idx, _ in top_synth]
        synth_values = [count for _, count in top_synth]
        
        bars2 = ax2.barh(range(len(synth_preds)), synth_values, color='orange', alpha=0.8)
        ax2.set_yticks(range(len(synth_preds)))
        ax2.set_yticklabels(synth_preds, fontsize=10)
        ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Top 20 Predicates - Synthetic Data', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predicate_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_classes_comparison(self, output_dir):
        """Plot side-by-side comparison of top classes"""
        # Get common classes
        real_classes = set(self.real_data['labels'])
        synth_classes = set(self.synthetic_data['labels'])
        common_classes = real_classes & synth_classes
        
        # Get top 15 common classes by total frequency
        class_totals = {}
        for cls in common_classes:
            real_count = self.real_data['label_counts'].get(cls, 0)
            synth_count = self.synthetic_data['label_counts'].get(cls, 0)
            class_totals[cls] = real_count + synth_count
        
        top_common = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)[:15]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        class_names = []
        real_values = []
        synth_values = []
        
        for cls, _ in top_common:
            cls_name = self.real_data['idx_to_classes'][cls] if cls < len(self.real_data['idx_to_classes']) else f"class_{cls}"
            class_names.append(cls_name)
            real_values.append(self.real_data['label_counts'].get(cls, 0))
            synth_values.append(self.synthetic_data['label_counts'].get(cls, 0))
        
        x_pos = np.arange(len(class_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, real_values, width, label='Real Data', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, synth_values, width, label='Synthetic Data', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_title('Top 15 Common Object Classes Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Object Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.legend(title='Dataset', title_fontsize=12, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_classes_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_predicates_comparison(self, output_dir):
        """Plot side-by-side comparison of all predicates"""
        # Get all predicates from real data (since it has more predicates)
        real_preds = set(self.real_data['predicates'])
        synth_preds = set(self.synthetic_data['predicates'])
        all_preds = real_preds | synth_preds  # Union of all predicates
        
        # Get all predicates sorted by total frequency
        pred_totals = {}
        for pred in all_preds:
            real_count = self.real_data['predicate_counts'].get(pred, 0)
            synth_count = self.synthetic_data['predicate_counts'].get(pred, 0)
            pred_totals[pred] = real_count + synth_count
        
        # Sort all predicates by total frequency
        all_preds_sorted = sorted(pred_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure with appropriate size based on number of predicates
        num_preds = len(all_preds_sorted)
        fig_height = max(12, num_preds * 0.3)  # Dynamic height based on number of predicates
        
        fig, ax = plt.subplots(figsize=(16, fig_height))
        
        pred_names = []
        real_values = []
        synth_values = []
        
        for pred, _ in all_preds_sorted:
            pred_name = self.real_data['idx_to_predicates'][pred] if pred < len(self.real_data['idx_to_predicates']) else f"pred_{pred}"
            pred_names.append(pred_name)
            real_values.append(self.real_data['predicate_counts'].get(pred, 0))
            synth_values.append(self.synthetic_data['predicate_counts'].get(pred, 0))
        
        # Use horizontal bar chart for better readability with many predicates
        y_pos = np.arange(len(pred_names))
        height = 0.35
        
        bars1 = ax.barh(y_pos - height/2, real_values, height, label='Real Data', 
                       color='lightgreen', alpha=0.8)
        bars2 = ax.barh(y_pos + height/2, synth_values, height, label='Synthetic Data', 
                       color='orange', alpha=0.8)
        
        ax.set_title(f'All Predicates Comparison ({num_preds} total)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicates', fontsize=12, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pred_names, fontsize=9)
        ax.legend(title='Dataset', title_fontsize=12, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add count annotations for better readability
        for i, (real_val, synth_val) in enumerate(zip(real_values, synth_values)):
            if real_val > 0:
                ax.text(real_val + max(real_values) * 0.01, i - height/2, str(real_val), 
                       va='center', ha='left', fontsize=8)
            if synth_val > 0:
                ax.text(synth_val + max(synth_values) * 0.01, i + height/2, str(synth_val), 
                       va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_predicates_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_statistics(self, output_dir):
        """Plot distribution statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Object class frequency distribution
        real_obj_freq = list(self.real_data['label_counts'].values())
        synth_obj_freq = list(self.synthetic_data['label_counts'].values())
        
        ax1.hist(real_obj_freq, bins=50, alpha=0.7, label='Real Data', color='skyblue', density=True)
        ax1.hist(synth_obj_freq, bins=50, alpha=0.7, label='Synthetic Data', color='lightcoral', density=True)
        ax1.set_xlabel('Class Frequency', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Object Class Frequency Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Predicate frequency distribution
        real_pred_freq = list(self.real_data['predicate_counts'].values())
        synth_pred_freq = list(self.synthetic_data['predicate_counts'].values())
        
        ax2.hist(real_pred_freq, bins=50, alpha=0.7, label='Real Data', color='lightgreen', density=True)
        ax2.hist(synth_pred_freq, bins=50, alpha=0.7, label='Synthetic Data', color='orange', density=True)
        ax2.set_xlabel('Predicate Frequency', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Predicate Frequency Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Class diversity comparison
        real_unique = len(set(self.real_data['labels']))
        synth_unique = len(set(self.synthetic_data['labels']))
        real_total = len(self.real_data['labels'])
        synth_total = len(self.synthetic_data['labels'])
        
        categories = ['Real Data', 'Synthetic Data']
        unique_counts = [real_unique, synth_unique]
        total_counts = [real_total, synth_total]
        
        x_pos = np.arange(len(categories))
        ax3.bar(x_pos, unique_counts, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax3.set_xlabel('Dataset', fontsize=12)
        ax3.set_ylabel('Unique Classes', fontsize=12)
        ax3.set_title('Object Class Diversity', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories)
        
        # Add total counts as text
        for i, (unique, total) in enumerate(zip(unique_counts, total_counts)):
            ax3.text(i, unique + max(unique_counts) * 0.01, f'Total: {total}', 
                    ha='center', va='bottom', fontsize=10)
        
        # Predicate diversity comparison
        real_pred_unique = len(set(self.real_data['predicates']))
        synth_pred_unique = len(set(self.synthetic_data['predicates']))
        real_pred_total = len(self.real_data['predicates'])
        synth_pred_total = len(self.synthetic_data['predicates'])
        
        pred_unique_counts = [real_pred_unique, synth_pred_unique]
        pred_total_counts = [real_pred_total, synth_pred_total]
        
        ax4.bar(x_pos, pred_unique_counts, color=['lightgreen', 'orange'], alpha=0.8)
        ax4.set_xlabel('Dataset', fontsize=12)
        ax4.set_ylabel('Unique Predicates', fontsize=12)
        ax4.set_title('Predicate Diversity', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        
        # Add total counts as text
        for i, (unique, total) in enumerate(zip(pred_unique_counts, pred_total_counts)):
            ax4.text(i, unique + max(pred_unique_counts) * 0.01, f'Total: {total}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir):
        """Generate a comprehensive summary report"""
        print(f"\n=== Generating Summary Report ===")
        
        report_path = os.path.join(output_dir, 'label_distribution_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("LABEL DISTRIBUTION COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Real Data (Visual Genome):\n")
            f.write(f"  - Total objects: {len(self.real_data['labels'])}\n")
            f.write(f"  - Total relations: {len(self.real_data['predicates'])}\n")
            f.write(f"  - Unique object classes: {len(set(self.real_data['labels']))}\n")
            f.write(f"  - Unique predicates: {len(set(self.real_data['predicates']))}\n\n")
            
            f.write(f"Synthetic Data:\n")
            f.write(f"  - Total objects: {len(self.synthetic_data['labels'])}\n")
            f.write(f"  - Total relations: {len(self.synthetic_data['predicates'])}\n")
            f.write(f"  - Unique object classes: {len(set(self.synthetic_data['labels']))}\n")
            f.write(f"  - Unique predicates: {len(set(self.synthetic_data['predicates']))}\n\n")
            
            # Object class analysis
            obj_analysis = self.analyze_object_class_distribution()
            f.write("OBJECT CLASS ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Common classes: {len(obj_analysis['real_classes'] & obj_analysis['synth_classes'])}\n")
            f.write(f"Real-only classes: {len(obj_analysis['real_classes'] - obj_analysis['synth_classes'])}\n")
            f.write(f"Synthetic-only classes: {len(obj_analysis['synth_classes'] - obj_analysis['real_classes'])}\n\n")
            
            f.write("Top 10 classes with significant frequency differences:\n")
            for cls_name, real_pct, synth_pct, diff_pct in obj_analysis['significant_diffs'][:10]:
                f.write(f"  {cls_name}: Real={real_pct:.3f}, Synthetic={synth_pct:.3f}, Diff={diff_pct:.3f}\n")
            f.write("\n")
            
            # Predicate analysis
            pred_analysis = self.analyze_predicate_distribution()
            f.write("PREDICATE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Common predicates: {len(pred_analysis['real_predicates'] & pred_analysis['synth_predicates'])}\n")
            f.write(f"Real-only predicates: {len(pred_analysis['real_predicates'] - pred_analysis['synth_predicates'])}\n")
            f.write(f"Synthetic-only predicates: {len(pred_analysis['synth_predicates'] - pred_analysis['real_predicates'])}\n\n")
            
            f.write("Top 10 predicates with significant frequency differences:\n")
            for pred_name, real_pct, synth_pct, diff_pct in pred_analysis['significant_diffs'][:10]:
                f.write(f"  {pred_name}: Real={real_pct:.4f}, Synthetic={synth_pct:.4f}, Diff={diff_pct:.4f}\n")
            f.write("\n")
            
            # Statistical tests
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 22 + "\n")
            
            # Chi-square test for object classes
            common_obj_classes = obj_analysis['real_classes'] & obj_analysis['synth_classes']
            if len(common_obj_classes) > 0:
                real_obj_counts = [self.real_data['label_counts'].get(cls, 0) for cls in common_obj_classes]
                synth_obj_counts = [self.synthetic_data['label_counts'].get(cls, 0) for cls in common_obj_classes]
                
                try:
                    chi2_obj, p_val_obj = stats.chisquare(real_obj_counts, synth_obj_counts)
                    f.write(f"Chi-square test for object classes: χ²={chi2_obj:.2f}, p={p_val_obj:.2e}\n")
                except:
                    f.write("Chi-square test for object classes: Could not compute\n")
            
            # Chi-square test for predicates
            common_predicates = pred_analysis['real_predicates'] & pred_analysis['synth_predicates']
            if len(common_predicates) > 0:
                real_pred_counts = [self.real_data['predicate_counts'].get(pred, 0) for pred in common_predicates]
                synth_pred_counts = [self.synthetic_data['predicate_counts'].get(pred, 0) for pred in common_predicates]
                
                try:
                    chi2_pred, p_val_pred = stats.chisquare(real_pred_counts, synth_pred_counts)
                    f.write(f"Chi-square test for predicates: χ²={chi2_pred:.2f}, p={p_val_pred:.2e}\n")
                except:
                    f.write("Chi-square test for predicates: Could not compute\n")
            
            f.write("\n")
            f.write("CONCLUSIONS\n")
            f.write("-" * 12 + "\n")
            f.write("1. The analysis reveals differences in label distributions between real and synthetic data.\n")
            f.write("2. Both datasets share most common classes and predicates, indicating good coverage.\n")
            f.write("3. Frequency differences suggest potential domain adaptation challenges.\n")
            f.write("4. Consider balancing strategies for underrepresented classes/predicates.\n")
        
        print(f"✓ Summary report saved to: {report_path}")
    
    def run_analysis(self, output_dir='label_distribution_analysis'):
        """Run complete analysis"""
        print("=== Label Distribution Comparison Analysis ===")
        print(f"Config file: {self.config_file}")
        print(f"Output directory: {output_dir}")
        
        # Load data
        self.load_real_data()
        self.load_synthetic_data()
        
        # Run analyses
        print("\n" + "="*60)
        obj_analysis = self.analyze_object_class_distribution()
        pred_analysis = self.analyze_predicate_distribution()
        
        # Create visualizations
        print("\n" + "="*60)
        self.create_visualizations(output_dir)
        
        # Generate report
        print("\n" + "="*60)
        self.generate_summary_report(output_dir)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {output_dir}")
        print(f"Generated files:")
        print(f"  - object_class_distribution.png")
        print(f"  - predicate_distribution.png")
        print(f"  - top_classes_comparison.png")
        print(f"  - top_predicates_comparison.png")
        print(f"  - distribution_statistics.png")
        print(f"  - label_distribution_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Label distribution comparison analysis')
    parser.add_argument('--config-file', type=str,
                       default='/home/junhwanheo/SpeaQ/configs/speaq_multi_dataset.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output-dir', type=str, default='label_distribution_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = LabelDistributionAnalyzer(args.config_file)
    analyzer.run_analysis(args.output_dir)


if __name__ == "__main__":
    main()
