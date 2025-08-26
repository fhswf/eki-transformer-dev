"""
Detailed analysis script focusing on specific model size and quantization comparisons
as requested by the user.

Creates three specific graphs:
1. Grouped by model size: train/val loss vs quantization bit-width
2. Grouped by bit-width: train/val loss vs model size
3. Same as #2 but with hyperparameter separation

Fixed to exclude fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width calculations.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict, Counter

def extract_detailed_config_data(conn):
    """Extract detailed configuration and performance data"""
    runs_df = pd.read_sql_query("""
        SELECT run_id, name, config, summary, state
        FROM runs
    """, conn)
    
    # Get final training and validation losses for each run
    final_metrics = {}
    
    # Get final training loss (last recorded value)
    train_loss_query = """
        SELECT run_id, metric_value as final_train_loss
        FROM metrics m1
        WHERE metric_name = 'train/loss' 
        AND step = (
            SELECT MAX(step) 
            FROM metrics m2 
            WHERE m2.run_id = m1.run_id AND m2.metric_name = 'train/loss'
        )
    """
    train_losses = pd.read_sql_query(train_loss_query, conn)
    for _, row in train_losses.iterrows():
        final_metrics[row['run_id']] = {'train_loss': row['final_train_loss']}
    
    # Get validation loss (usually only one value per run)
    val_loss_query = """
        SELECT run_id, metric_value as final_val_loss
        FROM metrics
        WHERE metric_name = 'validate/loss'
    """
    val_losses = pd.read_sql_query(val_loss_query, conn)
    for _, row in val_losses.iterrows():
        if row['run_id'] in final_metrics:
            final_metrics[row['run_id']]['val_loss'] = row['final_val_loss']
        else:
            final_metrics[row['run_id']] = {'val_loss': row['final_val_loss']}
    
    detailed_data = []
    
    for _, row in runs_df.iterrows():
        try:
            config = json.loads(row['config'])
            summary = json.loads(row['summary'])
            
            # Extract model parameters
            model_config = config.get('model', {}).get('args', {})
            quant_config = config.get('quantization', {})
            optim_config = config.get('optim', {}).get('args', {})
            dataset_config = config.get('dataset', {}).get('dataloader', {})
            
            # Calculate model size
            n_embd = model_config.get('n_embd', 0)
            n_layer = model_config.get('n_layer', 0)
            n_head = model_config.get('n_head', 0)
            vocab_size = model_config.get('vocab_size', 0)
            
            # Estimate model parameters
            embed_params = vocab_size * n_embd
            layer_params = n_layer * (4 * n_embd * n_embd + 2 * n_embd)
            total_params = embed_params + layer_params
            
            # Extract quantization bit widths (excluding fixed 8-bit layers)
            bit_widths = []
            excluded_layers = ['transformer.emb_add', 'linear_out']
            
            if 'model' in quant_config and 'layers' in quant_config['model']:
                for layer_name, layer_config in quant_config['model']['layers'].items():
                    # Skip fixed 8-bit layers
                    if any(excluded in layer_name for excluded in excluded_layers):
                        continue
                        
                    if 'quantizers' in layer_config:
                        for quant_name, quant_config_item in layer_config['quantizers'].items():
                            if isinstance(quant_config_item, dict) and 'args' in quant_config_item:
                                bit_width = quant_config_item['args'].get('bit_width')
                                if bit_width:
                                    bit_widths.append(bit_width)
            
            # Determine the primary quantization bit width (should be uniform for variable layers)
            uniform_bit_width = None
            primary_bit_width = None
            if bit_widths:
                # Get the most common bit width (excluding 8-bit fixed layers)
                bit_width_counts = Counter(bit_widths)
                primary_bit_width = bit_width_counts.most_common(1)[0][0]
                
                # Check if all variable layers use the same bit width
                variable_bit_widths = [bw for bw in bit_widths if bw != 8]
                if variable_bit_widths and len(set(variable_bit_widths)) == 1:
                    uniform_bit_width = variable_bit_widths[0]
                elif variable_bit_widths:
                    uniform_bit_width = primary_bit_width  # Use most common bit width
            
            # Get final losses
            metrics = final_metrics.get(row['run_id'], {})
            
            data_row = {
                'run_id': row['run_id'],
                'name': row['name'],
                'state': row['state'],
                'n_embd': n_embd,
                'n_layer': n_layer,
                'n_head': n_head,
                'vocab_size': vocab_size,
                'total_params': total_params,
                'model_size_category': f"{n_embd}emb_{n_layer}L_{n_head}H",
                'quant_enabled': quant_config.get('quantize', False),
                'quant_type': quant_config.get('type', 'None'),
                'uniform_bit_width': uniform_bit_width,
                'primary_bit_width': primary_bit_width,
                'bit_width_category': f"{int(uniform_bit_width)}bit" if uniform_bit_width else "No quant",
                'all_bit_widths': bit_widths,
                'variable_layer_count': len([bw for bw in bit_widths if bw != 8]),
                'final_train_loss': metrics.get('train_loss'),
                'final_val_loss': metrics.get('val_loss'),
                'learning_rate': optim_config.get('learning_rate'),
                'batch_size': dataset_config.get('batch_size'),
                'weight_decay': optim_config.get('weight_decay', 0),
                'dropout': model_config.get('dropout', 0),
                'hyperparams_signature': f"lr{optim_config.get('learning_rate', 0)}_bs{dataset_config.get('batch_size', 0)}_wd{optim_config.get('weight_decay', 0)}_do{model_config.get('dropout', 0)}"
            }
            detailed_data.append(data_row)
            
        except Exception as e:
            print(f"Error parsing config for run {row['run_id']}: {e}")
    
    return pd.DataFrame(detailed_data)
    
    detailed_data = []
    
    for _, row in runs_df.iterrows():
        try:
            config = json.loads(row['config'])
            summary = json.loads(row['summary'])
            
            # Extract model parameters
            model_config = config.get('model', {}).get('args', {})
            quant_config = config.get('quantization', {})
            optim_config = config.get('optim', {}).get('args', {})
            dataset_config = config.get('dataset', {}).get('dataloader', {})
            
            # Calculate model size
            n_embd = model_config.get('n_embd', 0)
            n_layer = model_config.get('n_layer', 0)
            n_head = model_config.get('n_head', 0)
            vocab_size = model_config.get('vocab_size', 0)
            
            # Estimate model parameters
            embed_params = vocab_size * n_embd
            layer_params = n_layer * (4 * n_embd * n_embd + 2 * n_embd)
            total_params = embed_params + layer_params
            
            # Extract quantization bit widths (excluding fixed 8-bit layers)
            bit_widths = []
            excluded_layers = ['transformer.emb_add', 'linear_out']
            
            if 'model' in quant_config and 'layers' in quant_config['model']:
                for layer_name, layer_config in quant_config['model']['layers'].items():
                    # Skip fixed 8-bit layers
                    if any(excluded in layer_name for excluded in excluded_layers):
                        continue
                        
                    if 'quantizers' in layer_config:
                        for quant_name, quant_config_item in layer_config['quantizers'].items():
                            if isinstance(quant_config_item, dict) and 'args' in quant_config_item:
                                bit_width = quant_config_item['args'].get('bit_width')
                                if bit_width:
                                    bit_widths.append(bit_width)
            
            # Determine the primary quantization bit width (should be uniform for variable layers)
            uniform_bit_width = None
            primary_bit_width = None
            if bit_widths:
                # Get the most common bit width (excluding 8-bit fixed layers)
                bit_width_counts = Counter(bit_widths)
                primary_bit_width = bit_width_counts.most_common(1)[0][0]
                
                # Check if all variable layers use the same bit width
                variable_bit_widths = [bw for bw in bit_widths if bw != 8]
                if variable_bit_widths and len(set(variable_bit_widths)) == 1:
                    uniform_bit_width = variable_bit_widths[0]
                elif variable_bit_widths:
                    uniform_bit_width = primary_bit_width  # Use most common bit width
            
            # Get final losses
            metrics = final_metrics.get(row['run_id'], {})
            
            data_row = {
                'run_id': row['run_id'],
                'name': row['name'],
                'state': row['state'],
                'n_embd': n_embd,
                'n_layer': n_layer,
                'n_head': n_head,
                'vocab_size': vocab_size,
                'total_params': total_params,
                'model_size_category': f"{n_embd}emb_{n_layer}L_{n_head}H",
                'quant_enabled': quant_config.get('quantize', False),
                'quant_type': quant_config.get('type', 'None'),
                'uniform_bit_width': uniform_bit_width,
                'primary_bit_width': primary_bit_width,
                'bit_width_category': f"{int(uniform_bit_width)}bit" if uniform_bit_width else "No quant",
                'all_bit_widths': bit_widths,
                'variable_layer_count': len([bw for bw in bit_widths if bw != 8]),
                'final_train_loss': metrics.get('train_loss'),
                'final_val_loss': metrics.get('val_loss'),
                'learning_rate': optim_config.get('learning_rate'),
                'batch_size': dataset_config.get('batch_size'),
                'weight_decay': optim_config.get('weight_decay', 0),
                'dropout': model_config.get('dropout', 0),
                'hyperparams_signature': f"lr{optim_config.get('learning_rate', 0)}_bs{dataset_config.get('batch_size', 0)}_wd{optim_config.get('weight_decay', 0)}_do{model_config.get('dropout', 0)}"
            }
            detailed_data.append(data_row)
            
        except Exception as e:
            print(f"Error parsing config for run {row['run_id']}: {e}")
    
    return pd.DataFrame(detailed_data)

def plot_loss_vs_bitwidth_by_model_size(df, training_curves_df):
    """Graph 1: Grouped by model size, plot training loss curves vs quantization bit-width"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs bit-width analysis")
        return
    
    # Group by model size
    model_sizes = valid_df['model_size_category'].unique()
    
    fig, axes = plt.subplots(1, len(model_sizes) if len(model_sizes) > 1 else 2, figsize=(16, 6))
    if len(model_sizes) == 1:
        axes = [axes[0], axes[1]]  # Ensure we have at least 2 subplots
    
    fig.suptitle('Training Loss Curves vs Quantization Bit-Width (Grouped by Model Size)', fontsize=14)
    
    # Get unique bit widths for color mapping
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot 1: Training curves colored by bit-width
    ax1 = axes[0]
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty:
            color = bit_width_colors[row['uniform_bit_width']]
            label = f"{int(row['uniform_bit_width'])}bit ({row['model_size_category']})"
            ax1.plot(run_curves['step'], run_curves['metric_value'], 
                    color=color, label=label, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves by Quantization Bit-Width')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final loss vs bit-width scatter
    ax2 = axes[1]
    for i, model_size in enumerate(model_sizes):
        size_data = valid_df[valid_df['model_size_category'] == model_size]
        ax2.scatter(size_data['uniform_bit_width'], size_data['final_train_loss'], 
                   s=150, alpha=0.8, label=model_size)
        
        # Add trend line if more than 1 point
        if len(size_data) > 1:
            z = np.polyfit(size_data['uniform_bit_width'], size_data['final_train_loss'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(size_data['uniform_bit_width'].min(), size_data['uniform_bit_width'].max(), 100)
            ax2.plot(x_trend, p(x_trend), linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Quantization Bit-Width (Variable Layers)')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Final Training Loss vs Bit-Width')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations with bit widths
    for _, row in valid_df.iterrows():
        ax2.annotate(f'{int(row["uniform_bit_width"])}bit', 
                    (row['uniform_bit_width'], row['final_train_loss']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_curves_vs_bitwidth_by_model_size.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training curves vs bit-width (by model size) to: training_curves_vs_bitwidth_by_model_size.png")
    plt.show()

def plot_loss_vs_model_size_by_bitwidth(df, training_curves_df):
    """Graph 2: Grouped by bit-width, plot training loss curves vs model size"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['total_params', 'final_train_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs model size analysis")
        return
    
    # Group by bit width
    bit_widths = sorted(valid_df['uniform_bit_width'].dropna().unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Curves vs Model Size (Grouped by Bit-Width)', fontsize=14)
    
    # Get colors for each bit width
    colors = plt.cm.Set2(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot 1: Training curves colored by bit-width
    ax1 = axes[0]
    for _, row in valid_df.iterrows():
        if pd.notna(row['uniform_bit_width']):
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                color = bit_width_colors[row['uniform_bit_width']]
                label = f"{int(row['uniform_bit_width'])}bit ({row['total_params']/1e6:.1f}M params)"
                ax1.plot(run_curves['step'], run_curves['metric_value'], 
                        color=color, label=label, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves by Quantization Level')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final loss vs model size scatter (grouped by bit-width)
    ax2 = axes[1]
    for i, bit_width in enumerate(bit_widths):
        bw_data = valid_df[valid_df['uniform_bit_width'] == bit_width]
        if not bw_data.empty:
            ax2.scatter(bw_data['total_params'], bw_data['final_train_loss'], 
                       color=colors[i], label=f"{int(bit_width)}bit", s=150, alpha=0.8)
    
    ax2.set_xlabel('Model Size (Parameters)')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Final Training Loss vs Model Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add parameter count annotations
    for _, row in valid_df.iterrows():
        if pd.notna(row['uniform_bit_width']):
            ax2.annotate(f'{row["total_params"]/1e6:.1f}M\n{int(row["uniform_bit_width"])}bit', 
                        (row['total_params'], row['final_train_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, 
                        ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_curves_vs_model_size_by_bitwidth.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training curves vs model size (by bit-width) to: training_curves_vs_model_size_by_bitwidth.png")
    plt.show()

def plot_loss_vs_model_size_by_hyperparams(df, training_curves_df):
    """Graph 3: Training curves with hyperparameter and bit-width analysis"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['total_params', 'final_train_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs model size with hyperparameters analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Loss Analysis: Curves, Model Size, and Hyperparameters', fontsize=16)
    
    # Plot 1: Training curves colored by bit-width
    ax1 = axes[0, 0]
    bit_widths = sorted(valid_df['uniform_bit_width'].dropna().unique())
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors1[i] for i, bw in enumerate(bit_widths)}
    
    for _, row in valid_df.iterrows():
        if pd.notna(row['uniform_bit_width']):
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                color = bit_width_colors[row['uniform_bit_width']]
                label = f"{int(row['uniform_bit_width'])}bit (wd={row['weight_decay']}, do={row['dropout']})"
                ax1.plot(run_curves['step'], run_curves['metric_value'], 
                        color=color, label=label, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Curves by Quantization & Hyperparameters')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final loss vs bit-width with hyperparameter markers
    ax2 = axes[0, 1]
    # Create unique hyperparameter combinations
    hyperparam_combos = valid_df['hyperparams_signature'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    hyperparam_markers = {combo: markers[i % len(markers)] for i, combo in enumerate(hyperparam_combos)}
    
    for combo in hyperparam_combos:
        combo_data = valid_df[valid_df['hyperparams_signature'] == combo]
        if not combo_data.empty:
            # Extract readable hyperparams
            sample_row = combo_data.iloc[0]
            label = f"wd={sample_row['weight_decay']}, do={sample_row['dropout']}"
            
            ax2.scatter(combo_data['uniform_bit_width'], combo_data['final_train_loss'], 
                       marker=hyperparam_markers[combo], s=150, alpha=0.8, label=label)
    
    ax2.set_xlabel('Quantization Bit-Width')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Final Loss vs Bit-Width by Hyperparameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hyperparameter sensitivity analysis
    ax3 = axes[1, 0]
    # Weight decay vs performance, colored by bit-width
    scatter = ax3.scatter(valid_df['weight_decay'], valid_df['final_train_loss'], 
                         c=valid_df['uniform_bit_width'], cmap='plasma', s=150, alpha=0.8)
    ax3.set_xlabel('Weight Decay')
    ax3.set_ylabel('Final Training Loss')
    ax3.set_title('Training Loss vs Weight Decay (colored by bit-width)')
    cbar3 = plt.colorbar(scatter, ax=ax3)
    cbar3.set_label('Bit Width (Variable Layers)')
    ax3.grid(True, alpha=0.3)
    
    # Add annotations for each point
    for _, row in valid_df.iterrows():
        if pd.notna(row['uniform_bit_width']):
            ax3.annotate(f'{int(row["uniform_bit_width"])}bit\ndo={row["dropout"]}', 
                        (row['weight_decay'], row['final_train_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        ha='left', va='bottom')
    
    # Plot 4: Convergence analysis - steps to reach certain loss thresholds
    ax4 = axes[1, 1]
    target_losses = [6.0, 5.0, 4.5, 4.0]  # Loss thresholds to analyze
    
    convergence_data = []
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty and pd.notna(row['uniform_bit_width']):
            for target_loss in target_losses:
                # Find first step where loss drops below target
                below_target = run_curves[run_curves['metric_value'] <= target_loss]
                if not below_target.empty:
                    first_step = below_target['step'].iloc[0]
                    convergence_data.append({
                        'bit_width': int(row['uniform_bit_width']),
                        'target_loss': target_loss,
                        'steps_to_reach': first_step,
                        'weight_decay': row['weight_decay'],
                        'dropout': row['dropout']
                    })
    
    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        for target_loss in target_losses:
            target_data = conv_df[conv_df['target_loss'] == target_loss]
            if not target_data.empty:
                ax4.scatter(target_data['bit_width'], target_data['steps_to_reach'], 
                           label=f'Loss<{target_loss}', s=100, alpha=0.7)
        
        ax4.set_xlabel('Quantization Bit-Width')
        ax4.set_ylabel('Steps to Reach Target Loss')
        ax4.set_title('Convergence Speed vs Quantization Level')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor convergence analysis', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Convergence Analysis (No Data)')
    
    plt.tight_layout()
    plt.savefig('training_curves_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved comprehensive training curves analysis to: training_curves_comprehensive_analysis.png")
    plt.show()

def create_data_summary_table(df):
    """Create a summary table of the analyzed data"""
    print("\n" + "="*80)
    print("DETAILED DATA SUMMARY")
    print("="*80)
    
    print(f"\nTotal Runs Analyzed: {len(df)}")
    print(f"Runs with Training Loss: {df['final_train_loss'].notna().sum()}")
    print(f"Runs with Validation Loss: {df['final_val_loss'].notna().sum()}")
    print(f"Runs with Quantization: {df['quant_enabled'].sum()}")
    
    print("\n--- Model Architecture Variations ---")
    print("Model Size Categories:")
    for size_cat in df['model_size_category'].unique():
        count = len(df[df['model_size_category'] == size_cat])
        params = df[df['model_size_category'] == size_cat]['total_params'].iloc[0]
        print(f"  {size_cat}: {count} runs ({params/1e6:.1f}M parameters)")
    
    print("\n--- Quantization Variations ---")
    print("Bit Width Categories (excluding fixed 8-bit layers):")
    for bw_cat in df['bit_width_category'].unique():
        count = len(df[df['bit_width_category'] == bw_cat])
        if 'bit' in bw_cat and bw_cat != "No quant":
            sample_row = df[df['bit_width_category'] == bw_cat].iloc[0]
            uniform_bw = sample_row['uniform_bit_width']
            primary_bw = sample_row['primary_bit_width']
            all_bws = sample_row['all_bit_widths']
            var_layers = sample_row['variable_layer_count']
            print(f"  {bw_cat}: {count} runs")
            print(f"    Primary bit-width: {int(primary_bw)} bits")
            print(f"    Variable layers: {var_layers}")
            print(f"    All bit-widths in config: {sorted(set(all_bws))}")
        else:
            print(f"  {bw_cat}: {count} runs")
    
    print("\n--- Hyperparameter Variations ---")
    print(f"Learning Rates: {sorted(df['learning_rate'].unique())}")
    print(f"Batch Sizes: {sorted(df['batch_size'].unique())}")
    print(f"Weight Decay: {sorted(df['weight_decay'].unique())}")
    print(f"Dropout: {sorted(df['dropout'].unique())}")
    
    print("\n--- Performance Summary ---")
    if df['final_train_loss'].notna().any():
        print(f"Training Loss Range: {df['final_train_loss'].min():.4f} - {df['final_train_loss'].max():.4f}")
    if df['final_val_loss'].notna().any():
        print(f"Validation Loss Range: {df['final_val_loss'].min():.4f} - {df['final_val_loss'].max():.4f}")

def main():
    """Main analysis function"""
    print("🔍 Creating detailed model size and quantization analysis with training curves...")
    print("Note: Excluding fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width analysis")
    
    conn = sqlite3.connect('wandb_data.db')
    
    # Extract detailed configuration data
    print("\n1. Extracting detailed configuration data...")
    df = extract_detailed_config_data(conn)
    
    # Get training curves for all runs
    print("2. Extracting training curves...")
    training_curves_df = get_training_curves(conn)
    print(f"   Found {len(training_curves_df)} training data points across all runs")
    
    # Create data summary
    create_data_summary_table(df)
    
    # Create the three requested graphs with training curves
    print("\n3. Creating Graph 1: Training Curves vs Bit-Width (grouped by Model Size)...")
    plot_loss_vs_bitwidth_by_model_size(df, training_curves_df)
    
    print("\n4. Creating Graph 2: Training Curves vs Model Size (grouped by Bit-Width)...")
    plot_loss_vs_model_size_by_bitwidth(df, training_curves_df)
    
    print("\n5. Creating Graph 3: Comprehensive Training Analysis with Hyperparameters...")
    plot_loss_vs_model_size_by_hyperparams(df, training_curves_df)
    
    # Save detailed data for further analysis
    df.to_csv('detailed_analysis_data.csv', index=False)
    training_curves_df.to_csv('training_curves_data.csv', index=False)
    print("\n💾 Saved detailed analysis data to: detailed_analysis_data.csv")
    print("💾 Saved training curves data to: training_curves_data.csv")
    
    conn.close()
    print("\n✅ Detailed training curves analysis complete!")

if __name__ == "__main__":
    main()
