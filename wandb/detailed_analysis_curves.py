"""
Detailed analysis script focusing on training curves over time instead of just final values.
Creates three specific graphs with training loss progression:
1. Training curves vs quantization bit-width (grouped by model size)
2. Training curves vs model size (grouped by bit-width)  
3. Comprehensive analysis with hyperparameters and convergence

Fixed to exclude fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width calculations.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
import sys
from collections import defaultdict, Counter


def calculate_transformer_param_count(n_layers, d_model, vocab_size, max_seq_length, expansion_factor=4):
    # Multi-head self-attention
    n_atte = (4 * (d_model**2 + d_model))
    # Feedforward network
    n_ff = (expansion_factor * d_model**2 + d_model**2 * expansion_factor)
    # LayerNorm parameters (two LayerNorm layers per transformer block)
    n_ln = 2 * d_model
    # decoder
    n_deco = n_atte + n_ff + 2*n_ln
    
    # Embedding parameters: vocab embeddings + position embeddings
    n_emb_in = (vocab_size * d_model)  # + (d_model * max_seq_length)
    n_emb_out = vocab_size * d_model
    n_emb = n_emb_in + n_emb_out
    
    total_params = n_layers * n_deco + n_emb + n_ln 
    
    return total_params

def extract_detailed_config_data(conn, use_fixed_quant_for_all_layers=True):
    """Extract detailed configuration and performance data for ALL runs"""
    # Get ALL runs instead of sampling
    runs_df = pd.read_sql_query(f"""
        SELECT run_id, name, config, summary, state
        FROM runs
        WHERE state IN ('finished', 'running', 'crashed')
        ORDER BY name
    """, conn)
    
    print(f"   Processing ALL {len(runs_df)} runs ...")
    
    # OPTIMIZED: Get final training losses efficiently using window functions
    final_metrics = {}
    
    if not runs_df.empty:
        run_ids = "', '".join(runs_df['run_id'].tolist())
        
        # MUCH BETTER: Single query using window function to get last loss per run
        train_loss_query = f"""
            WITH ranked_metrics AS (
                SELECT run_id, metric_value as final_train_loss,
                       ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY step DESC) as rn
                FROM metrics
                WHERE metric_name = 'train/loss' 
                AND run_id IN ('{run_ids}')
            )
            SELECT run_id, final_train_loss
            FROM ranked_metrics
            WHERE rn = 1
        """
        train_losses = pd.read_sql_query(train_loss_query, conn)
        for _, row in train_losses.iterrows():
            final_metrics[row['run_id']] = {'train_loss': row['final_train_loss']}
        
        print(f"   Got final losses for {len(train_losses)} runs efficiently")
    
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
            
            # Extract normalization layer information
            norm_layer = model_config.get('norm_layer', 'LayerNorm')  # Default to LayerNorm
            
            # Calculate model size
            n_embd = model_config.get('n_embd', 0)
            n_layer = model_config.get('n_layer', 0)
            n_head = model_config.get('n_head', 0)
            vocab_size = model_config.get('vocab_size', 0)
            se = model_config.get('block_size', 0)

            total_params = calculate_transformer_param_count(n_layer, n_embd, vocab_size, se)
            
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
                                if use_fixed_quant_for_all_layers:  # stop and break at first sight
                                    break
            
            # Determine the primary quantization bit width
            if not use_fixed_quant_for_all_layers:
                # More complex logic for mixed quantization (currently unused)
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
            else:
                # Simple case: all layers use the same quantization
                uniform_bit_width = bit_widths[0] if bit_widths else None
                primary_bit_width = uniform_bit_width
            
            # Get final losses
            metrics = final_metrics.get(row['run_id'], {})
            
            data_row = {
                'run_id': row['run_id'],
                'name': row['name'],
                'n_embd': n_embd,
                'n_layer': n_layer,
                'n_head': n_head,
                'vocab_size': vocab_size,
                'total_params': total_params,
                'model_size_category': f"{n_embd}emb_{n_layer}L_{n_head}H",
                'norm_layer': norm_layer,
                'quant_enabled': quant_config.get('quantize', False),
                'quant_type': quant_config.get('type', 'None'),
                'uniform_bit_width': uniform_bit_width,
                'primary_bit_width': primary_bit_width,
                'bit_width_category': f"{int(uniform_bit_width)}bit" if uniform_bit_width else "No quant",
                'final_train_loss': metrics.get('train_loss'),
                'final_val_loss': None,  # Skip validation loss for performance
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

def get_training_curves(conn, max_steps=400, min_steps=100, loss_type='train/loss'):
    """Extract training curves with all steps, filtering out short runs - DETERMINISTIC"""
    # Get ALL runs deterministically (no random sampling)
    all_runs_query = f"""
        SELECT DISTINCT run_id 
        FROM metrics 
        WHERE metric_name = '{loss_type}'
        ORDER BY run_id
    """
    all_runs_df = pd.read_sql_query(all_runs_query, conn)
    run_ids = "', '".join(all_runs_df['run_id'].tolist())
    
    # Get ALL steps for ALL runs (completely deterministic)
    training_curves_query = f"""
        SELECT run_id, step, metric_value
        FROM metrics
        WHERE metric_name = '{loss_type}'
        AND run_id IN ('{run_ids}')
        AND step <= {max_steps}
        ORDER BY run_id, step
    """
    print(f"   Getting ALL steps for ALL {len(all_runs_df)} runs (max {max_steps} steps) using {loss_type}...")
    result = pd.read_sql_query(training_curves_query, conn)
    
    # Filter out runs with too few steps
    run_step_counts = result.groupby('run_id')['step'].count()
    valid_runs = run_step_counts[run_step_counts >= min_steps].index.tolist()
    filtered_result = result[result['run_id'].isin(valid_runs)]
    
    print(f"   Filtered out {len(all_runs_df) - len(valid_runs)} short runs (< {min_steps} steps)")
    print(f"   Keeping ALL {len(valid_runs)} runs with {len(filtered_result)} total data points")
    
    return filtered_result

def plot_training_curves_by_model_size_subplots(df, training_curves_df):
    """Individual Plot 1a: Separate subplots for each model size, color-coded by bit width"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for training curves by model size analysis")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Get unique model sizes and bit widths
    model_sizes = sorted(valid_df['total_params'].unique())
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    
    # Create color mapping for bit widths
    colors = plt.cm.viridis(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Calculate subplot layout
    n_sizes = len(model_sizes)
    cols = min(3, n_sizes)  # Max 3 columns
    rows = (n_sizes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_sizes == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    total_curves = 0
    
    for idx, model_size in enumerate(model_sizes):
        ax = axes[idx] if n_sizes > 1 else axes[0]
        size_runs = valid_df[valid_df['total_params'] == model_size]
        curves_in_subplot = 0
        labeled_bits = set()
        
        for _, row in size_runs.iterrows():
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                bit_width = row['uniform_bit_width']
                color = bit_width_colors[bit_width]
                # Only show label for first curve of each bit width in this subplot
                label = f"{int(bit_width)}-bit" if bit_width not in labeled_bits else ""
                if bit_width not in labeled_bits:
                    labeled_bits.add(bit_width)
                
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linewidth=1.5, alpha=0.8, label=label)
                curves_in_subplot += 1
                total_curves += 1
        
        ax.set_title(f'Model Size: {model_size:,} params ({curves_in_subplot} curves)', fontsize=11)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Training Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
        if labeled_bits:
            ax.legend(fontsize=9, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n_sizes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Training Curves by Model Size (Color = Bit Width) - Total: {total_curves} curves', fontsize=14)
    plt.tight_layout()
    plt.savefig('1a_training_curves_by_model_size_subplots.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved training curves by model size subplots to: 1a_training_curves_by_model_size_subplots.png")
    plt.show()
    plt.close()


def plot_training_curves_by_bit_width_subplots(df, training_curves_df):
    """Individual Plot 1b: Separate subplots for each bit width, color-coded by model size"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for training curves by bit width analysis")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Get unique bit widths and model sizes
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    model_sizes = sorted(valid_df['total_params'].unique())
    
    # Create color mapping for model sizes
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_sizes)))
    model_size_colors = {size: colors[i] for i, size in enumerate(model_sizes)}
    
    # Calculate subplot layout
    n_bits = len(bit_widths)
    cols = min(3, n_bits)  # Max 3 columns
    rows = (n_bits + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_bits == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    total_curves = 0
    
    for idx, bit_width in enumerate(bit_widths):
        ax = axes[idx] if n_bits > 1 else axes[0]
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        curves_in_subplot = 0
        labeled_sizes = set()
        
        for _, row in bw_runs.iterrows():
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                model_size = row['total_params']
                color = model_size_colors[model_size]
                # Only show label for first curve of each model size in this subplot
                label = f"{model_size:,} params" if model_size not in labeled_sizes else ""
                if model_size not in labeled_sizes:
                    labeled_sizes.add(model_size)
                
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linewidth=1.5, alpha=0.8, label=label)
                curves_in_subplot += 1
                total_curves += 1
        
        ax.set_title(f'Quantization: {int(bit_width)}-bit ({curves_in_subplot} curves)', fontsize=11)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Training Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
        if labeled_sizes:
            ax.legend(fontsize=9, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n_bits, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Training Curves by Bit Width (Color = Model Size) - Total: {total_curves} curves', fontsize=14)
    plt.tight_layout()
    plt.savefig('1b_training_curves_by_bit_width_subplots.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved training curves by bit width subplots to: 1b_training_curves_by_bit_width_subplots.png")
    plt.show()
    plt.close()


def plot_training_curves_by_model_size(df, training_curves_df):
    """Individual Plot 1a: Training curves grouped by model size"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for training curves by model size analysis")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get unique model sizes for grouping
    model_sizes = sorted(valid_df['total_params'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_sizes)))
    model_size_colors = {size: colors[i] for i, size in enumerate(model_sizes)}
    
    # Plot training curves grouped by model size
    plotted_curves = 0
    labeled_sizes = set()  # Track which model sizes have been labeled
    
    for model_size in model_sizes:
        size_runs = valid_df[valid_df['total_params'] == model_size]
        
        for _, row in size_runs.iterrows():
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                color = model_size_colors[model_size]
                # Only show label for first curve of each model size
                label = f"{model_size:,} params" if model_size not in labeled_sizes else ""
                if model_size not in labeled_sizes:
                    labeled_sizes.add(model_size)
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linewidth=2, alpha=0.7, label=label)
                plotted_curves += 1
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Training Loss Curves Grouped by Model Size (ALL {plotted_curves} curves)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1c_training_curves_by_model_size.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training curves grouped by model size to: 1c_training_curves_by_model_size.png")
    plt.show()
    plt.close()


def plot_training_curves_by_bit_width(df, training_curves_df):
    """Individual Plot 1b: Training curves grouped by bit width"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for training curves by bit width analysis")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get unique bit widths for grouping
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot training curves grouped by bit width
    plotted_curves = 0
    
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        curves_in_group = 0
        
        for _, row in bw_runs.iterrows():
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                color = bit_width_colors[bit_width]
                # Only show label for first curve of each bit width
                label = f"{int(bit_width)}-bit" if curves_in_group == 0 else ""
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linewidth=2, alpha=0.7, label=label)
                plotted_curves += 1
                curves_in_group += 1
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Training Loss Curves Grouped by Quantization Bit-Width (ALL {plotted_curves} curves)', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1d_training_curves_by_bit_width.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training curves grouped by bit width to: 1d_training_curves_by_bit_width.png")
    plt.show()
    plt.close()



def plot_average_convergence_curves(df, training_curves_df):
    """Individual Plot 3: Average training curves by quantization level"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for average convergence analysis")
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot average training curves
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        bw_run_ids = bw_runs['run_id'].tolist()
        bw_curves = training_curves_df[training_curves_df['run_id'].isin(bw_run_ids)]
        
        if not bw_curves.empty:
            # Calculate average curve for this bit width
            avg_curve = bw_curves.groupby('step')['metric_value'].mean().reset_index()
            std_curve = bw_curves.groupby('step')['metric_value'].std().reset_index()
            
            color = bit_width_colors[bit_width]
            label = f"{int(bit_width)}-bit (n={len(bw_runs)})"
            
            # Plot average with confidence interval
            ax.plot(avg_curve['step'], avg_curve['metric_value'], 
                   color=color, label=label, linewidth=3, alpha=0.9)
            
            # Add confidence interval if we have std data
            if not std_curve['metric_value'].isna().all():
                ax.fill_between(avg_curve['step'], 
                              avg_curve['metric_value'] - std_curve['metric_value'],
                              avg_curve['metric_value'] + std_curve['metric_value'],
                              color=color, alpha=0.2)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Training Loss', fontsize=12)
    ax.set_title('Average Training Curves by Quantization Bit-Width', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2_average_convergence_curves.png', dpi=150, bbox_inches='tight')
    print("📊 Saved average convergence curves to: 2_average_convergence_curves.png")
    plt.show()
    plt.close()


def plot_convergence_performance_summary(df):
    """Individual Plot 4: Convergence performance summary bar chart"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for convergence performance summary")
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Prepare convergence summary
    convergence_summary = []
    
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        final_losses = bw_runs['final_train_loss'].dropna()
        
        if not final_losses.empty:
            convergence_summary.append({
                'bit_width': int(bit_width),
                'mean_final_loss': final_losses.mean(),
                'std_final_loss': final_losses.std() if len(final_losses) > 1 else 0,
                'min_final_loss': final_losses.min(),
                'count': len(final_losses)
            })
    
    if convergence_summary:
        conv_df = pd.DataFrame(convergence_summary)
        bars = ax.bar(range(len(conv_df)), conv_df['mean_final_loss'], 
                     yerr=conv_df['std_final_loss'], capsize=10, alpha=0.8,
                     color=[bit_width_colors[bw/1] for bw in conv_df['bit_width']])
        
        ax.set_xlabel('Quantization Bit-Width', fontsize=12)
        ax.set_ylabel('Final Training Loss', fontsize=12)
        ax.set_title('Convergence Performance Summary by Bit-Width', fontsize=14)
        ax.set_xticks(range(len(conv_df)))
        ax.set_xticklabels([f'{bw}-bit' for bw in conv_df['bit_width']], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, row in conv_df.iterrows():
            ax.text(i, row['mean_final_loss'] + row['std_final_loss'] + 0.05,
                   f'{row["mean_final_loss"]:.3f}\\nn={row["count"]}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('3_convergence_performance_summary.png', dpi=150, bbox_inches='tight')
    print("📊 Saved convergence performance summary to: 3_convergence_performance_summary.png")
    plt.show()
    plt.close()


def plot_hyperparameter_curves_analysis(df, training_curves_df):
    """Individual Plot 5: Hyperparameter and quantization analysis colored by model size"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for hyperparameter analysis")
        return
    
    # Create single plot for hyperparameter landscape
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a performance landscape visualization colored by model size
    x_data = []
    y_data = []
    z_data = []  # This will now be model size for coloring
    performance_data = []  # Final loss for point size
    labels = []
    
    for _, row in valid_df.iterrows():
        x_data.append(row['uniform_bit_width'])
        y_data.append(row['weight_decay'] * 10 + row['dropout'])  # Combine hyperparams
        z_data.append(row['total_params'])  # Color by model size
        performance_data.append(row['final_train_loss'])  # Size by performance
        labels.append(f"wd={row['weight_decay']}, do={row['dropout']}")
    
    # Normalize performance for point sizes (smaller loss = larger point)
    min_loss = min(performance_data)
    max_loss = max(performance_data)
    normalized_perf = [(max_loss - loss + 0.1) / (max_loss - min_loss + 0.1) for loss in performance_data]
    point_sizes = [50 + 150 * perf for perf in normalized_perf]  # Scale point sizes
    
    scatter = ax.scatter(x_data, y_data, c=z_data, s=point_sizes, 
                        cmap='plasma', alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Quantization Bit-Width', fontsize=12)
    ax.set_ylabel('Hyperparameter Combination (wd*10 + dropout)', fontsize=12)
    ax.set_title('Performance Landscape: Quantization vs Hyperparameters\n(Color = Model Size, Size = Performance)', fontsize=14)
    
    # Colorbar for model size
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Model Size (Parameters)', fontsize=12)
    
    # Add annotations for better readability (limit to avoid clutter)
    for i, label in enumerate(labels[:15]):  # Show fewer to avoid clutter
        ax.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Add legend explanation for point sizes
    ax.text(0.02, 0.98, 'Point size: Better performance = Larger size', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved hyperparameter analysis colored by model size to: 4_hyperparameter_analysis.png")
    plt.show()
    plt.close()


def plot_training_efficiency_analysis(df, training_curves_df):
    """Individual Plot 8: Training efficiency analysis"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss'])
    
    if valid_df.empty:
        print("No valid data for training efficiency analysis")
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate training efficiency
    efficiency_data = []
    
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if len(run_curves) > 1:
            total_loss_reduction = run_curves['metric_value'].iloc[0] - run_curves['metric_value'].iloc[-1]
            total_steps = run_curves['step'].iloc[-1] - run_curves['step'].iloc[0]
            efficiency = total_loss_reduction / total_steps if total_steps > 0 else 0
            
            efficiency_data.append({
                'bit_width': row['uniform_bit_width'],
                'efficiency': efficiency,
                'weight_decay': row['weight_decay'],
                'dropout': row['dropout'],
                'final_loss': row['final_train_loss']
            })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        scatter = ax.scatter(eff_df['bit_width'], eff_df['efficiency'], 
                           c=eff_df['final_loss'], cmap='viridis', s=150, alpha=0.8)
        ax.set_xlabel('Quantization Bit-Width', fontsize=12)
        ax.set_ylabel('Training Efficiency (loss reduction per step)', fontsize=12)
        ax.set_title('Training Efficiency vs Quantization Bit-Width', fontsize=14)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Final Training Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient efficiency data', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('5_training_efficiency_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training efficiency analysis to: 5_training_efficiency_analysis.png")
    plt.show()
    plt.close()


def plot_detailed_norm_layers_by_hyperparams(df, training_curves_df, loss_type='train'):
    """Individual Plot 6: Detailed normalization analysis with separate plots for each hyperparameter combination"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['norm_layer', 'final_train_loss', 'total_params', 'batch_size', 'dropout', 'weight_decay', 'uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for detailed norm layers comparison")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Replace "BatchNormIdPure" with "BatchNorm" for better readability
    valid_df = valid_df.copy()
    valid_df['norm_layer'] = valid_df['norm_layer'].replace('BatchNormIdPure', 'BatchNorm')
    
    # Get unique values for each parameter
    unique_norm_types = sorted(valid_df['norm_layer'].unique())
    unique_batch_sizes = sorted(valid_df['batch_size'].unique())
    unique_dropouts = sorted(valid_df['dropout'].unique())
    unique_weight_decays = sorted(valid_df['weight_decay'].unique())
    unique_bit_widths = sorted(valid_df['uniform_bit_width'].dropna().unique())
    
    print(f"   Creating separate plots for each hyperparameter combination:")
    print(f"   - Norm types: {unique_norm_types}")
    print(f"   - Batch sizes: {unique_batch_sizes}")
    print(f"   - Dropouts: {unique_dropouts}")
    print(f"   - Weight decays: {unique_weight_decays}")
    print(f"   - Bit widths: {unique_bit_widths}")
    
    # Create combined color and style mapping for model size + bit width
    model_sizes = sorted(valid_df['total_params'].unique())
    
    # Use consistent COLORS for model sizes (easier to distinguish)
    model_size_colors = plt.cm.Set1(np.linspace(0, 1, len(model_sizes)))
    model_size_color_map = {size: model_size_colors[i] for i, size in enumerate(model_sizes)}
    
    # Use DISTINCT LINE STYLES for different bit widths (much more visible)
    bit_width_style_map = {
        2.0: ':',      # Dotted line for 2-bit
        3.0: '--',     # Dashed line for 3-bit  
        4.0: '-'       # Solid line for 4-bit
    }
    
    # Create comprehensive legend mapping
    def get_curve_properties(model_size, bit_width):
        # Color represents model size (consistent across all bit widths)
        color = model_size_color_map[model_size]
        
        # Line style represents bit width (consistent across all model sizes)
        linestyle = bit_width_style_map.get(bit_width, '-')  # Default to solid if unknown
        
        return color, linestyle
    
    # Calculate global y-axis limits for consistency across all plots
    all_training_data = []
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty:
            all_training_data.extend(run_curves['metric_value'].tolist())
    
    if all_training_data:
        global_y_min = min(all_training_data) * 0.95  # Add 5% padding
        global_y_max = max(all_training_data) * 1.05  # Add 5% padding
        print(f"   Using consistent y-axis range: [{global_y_min:.3f}, {global_y_max:.3f}]")
    else:
        global_y_min, global_y_max = None, None
    
    # 1. Create separate plots for each Batch Size + Norm Type combination
    plot_count = 0
    for batch_size in unique_batch_sizes:
        for norm_type in unique_norm_types:
            subset_data = valid_df[(valid_df['norm_layer'] == norm_type) & 
                                 (valid_df['batch_size'] == batch_size)]
            
            if subset_data.empty:
                continue
                
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            curves_plotted = 0
            labeled_combinations = set()
            
            for _, row in subset_data.iterrows():
                run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
                if not run_curves.empty:
                    model_size = row['total_params']
                    bit_width = row['uniform_bit_width']
                    color, linestyle = get_curve_properties(model_size, bit_width)
                    
                    # Create combined label with both model size and bit width
                    label_key = f"{model_size}_{bit_width}"
                    label = f"{model_size:,} params, {int(bit_width)}-bit" if label_key not in labeled_combinations else ""
                    if label_key not in labeled_combinations:
                        labeled_combinations.add(label_key)
                    
                    ax.plot(run_curves['step'], run_curves['metric_value'], 
                           color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8, label=label)
                    curves_plotted += 1
            
            ax.set_title(f'{norm_type} with Batch Size {batch_size} ({curves_plotted} curves)', fontsize=14, weight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            if global_y_min is not None and global_y_max is not None:
                ax.set_ylim(global_y_min, global_y_max)
            
            if labeled_combinations:
                ax.legend(fontsize=9, loc='upper right', ncol=1)
                
            # Add statistics
            if not subset_data.empty:
                final_losses = subset_data['final_train_loss'].dropna()
                if not final_losses.empty:
                    mean_loss = final_losses.mean()
                    std_loss = final_losses.std()
                    ax.text(0.02, 0.98, f'μ={mean_loss:.3f}\nσ={std_loss:.3f}', 
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            filename = f'7a_{norm_type}_batch{batch_size}_{loss_type}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Saved {filename}")
            # plt.show()  # Commented out for faster execution
            plt.close()
            plot_count += 1
    
    # 2. Create separate plots for each Dropout + Norm Type combination
    for dropout in unique_dropouts:
        for norm_type in unique_norm_types:
            subset_data = valid_df[(valid_df['norm_layer'] == norm_type) & 
                                 (valid_df['dropout'] == dropout)]
            
            if subset_data.empty:
                continue
                
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            curves_plotted = 0
            labeled_combinations = set()
            
            for _, row in subset_data.iterrows():
                run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
                if not run_curves.empty:
                    model_size = row['total_params']
                    bit_width = row['uniform_bit_width']
                    color, linestyle = get_curve_properties(model_size, bit_width)
                    
                    # Create combined label with both model size and bit width
                    label_key = f"{model_size}_{bit_width}"
                    label = f"{model_size:,} params, {int(bit_width)}-bit" if label_key not in labeled_combinations else ""
                    if label_key not in labeled_combinations:
                        labeled_combinations.add(label_key)
                    
                    ax.plot(run_curves['step'], run_curves['metric_value'], 
                           color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8, label=label)
                    curves_plotted += 1
            
            ax.set_title(f'{norm_type} with Dropout {dropout} ({curves_plotted} curves)', fontsize=14, weight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            if global_y_min is not None and global_y_max is not None:
                ax.set_ylim(global_y_min, global_y_max)
            
            if labeled_combinations:
                ax.legend(fontsize=9, loc='upper right', ncol=1)
                
            # Add statistics
            if not subset_data.empty:
                final_losses = subset_data['final_train_loss'].dropna()
                if not final_losses.empty:
                    mean_loss = final_losses.mean()
                    std_loss = final_losses.std()
                    ax.text(0.02, 0.98, f'μ={mean_loss:.3f}\nσ={std_loss:.3f}', 
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            filename = f'7b_{norm_type}_dropout{dropout}_{loss_type}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Saved {filename}")
            # plt.show()  # Commented out for faster execution
            plt.close()
            plot_count += 1
    
    # 3. Create separate plots for each Weight Decay + Norm Type combination
    for weight_decay in unique_weight_decays:
        for norm_type in unique_norm_types:
            subset_data = valid_df[(valid_df['norm_layer'] == norm_type) & 
                                 (valid_df['weight_decay'] == weight_decay)]
            
            if subset_data.empty:
                continue
                
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            curves_plotted = 0
            labeled_combinations = set()
            
            for _, row in subset_data.iterrows():
                run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
                if not run_curves.empty:
                    model_size = row['total_params']
                    bit_width = row['uniform_bit_width']
                    color, linestyle = get_curve_properties(model_size, bit_width)
                    
                    # Create combined label with both model size and bit width
                    label_key = f"{model_size}_{bit_width}"
                    label = f"{model_size:,} params, {int(bit_width)}-bit" if label_key not in labeled_combinations else ""
                    if label_key not in labeled_combinations:
                        labeled_combinations.add(label_key)
                    
                    ax.plot(run_curves['step'], run_curves['metric_value'], 
                           color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8, label=label)
                    curves_plotted += 1
            
            ax.set_title(f'{norm_type} with Weight Decay {weight_decay} ({curves_plotted} curves)', fontsize=14, weight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            if global_y_min is not None and global_y_max is not None:
                ax.set_ylim(global_y_min, global_y_max)
            
            if labeled_combinations:
                ax.legend(fontsize=9, loc='upper right', ncol=1)
                
            # Add statistics
            if not subset_data.empty:
                final_losses = subset_data['final_train_loss'].dropna()
                if not final_losses.empty:
                    mean_loss = final_losses.mean()
                    std_loss = final_losses.std()
                    ax.text(0.02, 0.98, f'μ={mean_loss:.3f}\nσ={std_loss:.3f}', 
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            filename = f'7c_{norm_type}_weightdecay{weight_decay}_{loss_type}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Saved {filename}")
            # plt.show()  # Commented out for faster execution
            plt.close()
            plot_count += 1
    
    print(f"📊 Generated {plot_count} individual detailed plots with consistent y-axis scaling and combined model size + bit-width encoding")


def plot_batch_size_comparison_by_norm_layers(df, training_curves_df, loss_type='train'):
    """Additional Plot: Compare batch sizes across all normalization layer types"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['norm_layer', 'final_train_loss', 'total_params', 'batch_size', 'uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for batch size comparison")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Replace "BatchNormIdPure" with "BatchNorm" for better readability
    valid_df = valid_df.copy()
    valid_df['norm_layer'] = valid_df['norm_layer'].replace('BatchNormIdPure', 'BatchNorm')
    
    # Get unique values
    unique_norm_types = sorted(valid_df['norm_layer'].unique())
    unique_batch_sizes = sorted(valid_df['batch_size'].unique())
    unique_bit_widths = sorted(valid_df['uniform_bit_width'].dropna().unique())
    
    print(f"   Creating batch size comparison across normalization types:")
    print(f"   - Norm types: {unique_norm_types}")
    print(f"   - Batch sizes: {unique_batch_sizes}")
    print(f"   - Bit widths: {unique_bit_widths}")
    
    # Create consistent color and style mapping
    # Colors for batch sizes (main distinction)
    batch_size_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_batch_sizes)))
    batch_size_color_map = {bs: batch_size_colors[i] for i, bs in enumerate(unique_batch_sizes)}
    
    # Line styles for bit widths (secondary distinction)
    bit_width_style_map = {
        2.0: ':',      # Dotted line for 2-bit
        3.0: '--',     # Dashed line for 3-bit  
        4.0: '-'       # Solid line for 4-bit
    }
    
    # Calculate global y-axis limits for consistency
    all_training_data = []
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty:
            all_training_data.extend(run_curves['metric_value'].tolist())
    
    if all_training_data:
        global_y_min = min(all_training_data) * 0.95
        global_y_max = max(all_training_data) * 1.05
    else:
        global_y_min, global_y_max = None, None
    
    # Create subplots: one for each normalization type
    n_norm_types = len(unique_norm_types)
    cols = min(2, n_norm_types)
    rows = (n_norm_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if n_norm_types == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    total_plots = 0
    
    for idx, norm_type in enumerate(unique_norm_types):
        ax = axes[idx] if n_norm_types > 1 else axes[0]
        norm_data = valid_df[valid_df['norm_layer'] == norm_type]
        
        curves_plotted = 0
        labeled_combinations = set()
        
        for _, row in norm_data.iterrows():
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                batch_size = row['batch_size']
                bit_width = row['uniform_bit_width']
                
                # Color by batch size, line style by bit width
                color = batch_size_color_map[batch_size]
                linestyle = bit_width_style_map.get(bit_width, '-')
                
                # Create combined label
                label_key = f"{batch_size}_{bit_width}"
                label = f"Batch {batch_size}, {int(bit_width)}-bit" if label_key not in labeled_combinations else ""
                if label_key not in labeled_combinations:
                    labeled_combinations.add(label_key)
                
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8, label=label)
                curves_plotted += 1
        
        ax.set_title(f'{norm_type} - Batch Size Comparison ({curves_plotted} curves)', 
                    fontsize=14, weight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        if global_y_min is not None and global_y_max is not None:
            ax.set_ylim(global_y_min, global_y_max)
        
        if labeled_combinations:
            ax.legend(fontsize=11, loc='upper right')
        
        # Add statistics
        final_losses_by_batch_bit = {}
        for batch_size in unique_batch_sizes:
            for bit_width in unique_bit_widths:
                batch_bit_data = norm_data[(norm_data['batch_size'] == batch_size) & 
                                         (norm_data['uniform_bit_width'] == bit_width)]
                if not batch_bit_data.empty:
                    final_losses = batch_bit_data['final_train_loss'].dropna()
                    if not final_losses.empty:
                        key = f"B{batch_size},{int(bit_width)}bit"
                        final_losses_by_batch_bit[key] = final_losses.mean()
        
        if final_losses_by_batch_bit:
            stats_text = '\n'.join([f'{key}: {loss:.3f}' 
                                  for key, loss in list(final_losses_by_batch_bit.items())[:4]])  # Limit to 4 entries
            ax.text(0.02, 0.98, f'Mean Final Loss:\n{stats_text}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        total_plots += 1
    
    # Hide unused subplots
    for idx in range(n_norm_types, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Batch Size Comparison Across Normalization Types\n(Color = Batch Size, Line Style = Bit Width)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'8_batch_size_comparison_by_norm_layers_{loss_type}.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved batch size comparison to: 8_batch_size_comparison_by_norm_layers_{loss_type}.png")
    # plt.show()  # Commented out for faster execution
    plt.close()
    
    # Create a summary bar chart showing mean performance by batch size and norm type
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    batch_norm_performance = []
    for norm_type in unique_norm_types:
        for batch_size in unique_batch_sizes:
            subset = valid_df[(valid_df['norm_layer'] == norm_type) & 
                            (valid_df['batch_size'] == batch_size)]
            if not subset.empty:
                final_losses = subset['final_train_loss'].dropna()
                if not final_losses.empty:
                    batch_norm_performance.append({
                        'norm_type': norm_type,
                        'batch_size': batch_size,
                        'mean_loss': final_losses.mean(),
                        'std_loss': final_losses.std() if len(final_losses) > 1 else 0,
                        'count': len(final_losses)
                    })
    
    if batch_norm_performance:
        perf_df = pd.DataFrame(batch_norm_performance)
        
        # Create grouped bar chart
        x = np.arange(len(unique_norm_types))
        width = 0.35
        
        for i, batch_size in enumerate(unique_batch_sizes):
            batch_data = perf_df[perf_df['batch_size'] == batch_size]
            values = []
            errors = []
            
            for norm_type in unique_norm_types:
                norm_batch_data = batch_data[batch_data['norm_type'] == norm_type]
                if not norm_batch_data.empty:
                    values.append(norm_batch_data['mean_loss'].iloc[0])
                    errors.append(norm_batch_data['std_loss'].iloc[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            color = batch_size_color_map[batch_size]
            bars = ax.bar(x + i * width - width/2, values, width, 
                         yerr=errors, capsize=5, alpha=0.8, 
                         color=color, label=f'Batch {batch_size}')
            
            # Add value labels on bars
            for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
                if val > 0:  # Only label non-zero bars
                    ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Normalization Type', fontsize=12)
        ax.set_ylabel('Mean Final Training Loss', fontsize=12)
        ax.set_title('Performance Comparison: Batch Size vs Normalization Type', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_norm_types)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'8b_batch_size_performance_summary_{loss_type}.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved batch size performance summary to: 8b_batch_size_performance_summary_{loss_type}.png")
    # plt.show()  # Commented out for faster execution
    plt.close()
    
    print(f"📊 Generated batch size comparison plots for {total_plots} normalization types")


def plot_compare_norm_layers(df, training_curves_df):
    """Individual Plot 6: Compare different normalization layer types and their training performance"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['norm_layer', 'final_train_loss', 'total_params'])
    
    if valid_df.empty:
        print("No valid data for norm layers comparison")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    # Replace "BatchNormIdPure" with "BatchNorm" for better readability
    valid_df = valid_df.copy()
    valid_df['norm_layer'] = valid_df['norm_layer'].replace('BatchNormIdPure', 'BatchNorm')
    
    # Show what normalization types we found
    norm_types = valid_df['norm_layer'].value_counts()
    print(f"   Found normalization types: {dict(norm_types)}")
    
    # Get unique norm types and model sizes
    unique_norm_types = sorted(valid_df['norm_layer'].unique())
    model_sizes = sorted(valid_df['total_params'].unique())
    
    # Create figure with subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color mappings
    norm_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_norm_types)))
    norm_type_colors = {norm: norm_colors[i] for i, norm in enumerate(unique_norm_types)}
    
    # Create color mapping for model sizes (for within-subplot color coding)
    model_size_colors = plt.cm.tab10(np.linspace(0, 1, len(model_sizes)))
    model_size_color_map = {size: model_size_colors[i] for i, size in enumerate(model_sizes)}
    
    # Calculate global y-axis limits for consistency across all plots
    all_training_data = []
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty:
            all_training_data.extend(run_curves['metric_value'].tolist())
    
    if all_training_data:
        global_y_min = min(all_training_data) * 0.95  # Add 5% padding
        global_y_max = max(all_training_data) * 1.05  # Add 5% padding
    else:
        global_y_min, global_y_max = None, None
    
    # 1-4. Training Curves by Normalization Type - One subplot per norm type
    axes = [ax1, ax2, ax3, ax4]
    
    for idx, norm_type in enumerate(unique_norm_types):
        if idx >= 4:  # Only show first 4 norm types
            break
            
        ax = axes[idx]
        norm_data = valid_df[valid_df['norm_layer'] == norm_type]
        curves_plotted = 0
        labeled_model_sizes = set()  # Track which model sizes have been labeled in this subplot
        
        for run_idx, (_, row) in enumerate(norm_data.iterrows()):
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                # Color by model size instead of norm type
                model_size = row['total_params']
                color = model_size_color_map[model_size]
                
                # Only show label for first curve of each model size in this subplot
                label = f"{model_size:,} params" if model_size not in labeled_model_sizes else ""
                if model_size not in labeled_model_sizes:
                    labeled_model_sizes.add(model_size)
                
                ax.plot(run_curves['step'], run_curves['metric_value'], 
                       color=color, linewidth=1.5, alpha=0.8, label=label)
                curves_plotted += 1
        
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Training Loss', fontsize=11)
        ax.set_title(f'{norm_type} ({curves_plotted} curves)', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        if global_y_min is not None and global_y_max is not None:
            ax.set_ylim(global_y_min, global_y_max)
        
        # Add legend for model sizes in this subplot
        if labeled_model_sizes:
            ax.legend(fontsize=9, loc='upper right')
        
        # Add statistics text box
        if not norm_data.empty:
            final_losses = norm_data['final_train_loss'].dropna()
            if not final_losses.empty:
                mean_loss = final_losses.mean()
                std_loss = final_losses.std()
                ax.text(0.02, 0.98, f'μ={mean_loss:.3f}\nσ={std_loss:.3f}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # If there are fewer than 4 norm types, hide unused subplots
    for idx in range(len(unique_norm_types), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training Curves by Normalization Type (Color = Model Size)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('6_normalization_layer_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved normalization layer analysis to: 6_normalization_layer_analysis.png")
    # plt.show()  # Commented out for faster execution
    plt.close()
    
    # Create a second figure for the performance comparison (keeping the bottom left graph you liked)
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Performance Comparison: Average Loss by Normalization Type (the graph you want to keep)
    norm_performance = []
    for norm_type in unique_norm_types:
        norm_data = valid_df[valid_df['norm_layer'] == norm_type]
        if not norm_data.empty:
            final_losses = norm_data['final_train_loss'].dropna()
            if not final_losses.empty:
                norm_performance.append({
                    'norm_type': norm_type,
                    'mean_loss': final_losses.mean(),
                    'std_loss': final_losses.std() if len(final_losses) > 1 else 0,
                    'count': len(final_losses)
                })
    
    if norm_performance:
        perf_df = pd.DataFrame(norm_performance)
        bars = ax.bar(range(len(perf_df)), perf_df['mean_loss'], 
                      yerr=perf_df['std_loss'], capsize=8, alpha=0.8,
                      color=[norm_type_colors[norm] for norm in perf_df['norm_type']])
        
        ax.set_xlabel('Normalization Type', fontsize=12)
        ax.set_ylabel('Mean Final Training Loss', fontsize=12)
        ax.set_title('Performance Comparison by Normalization Type', fontsize=14, weight='bold')
        ax.set_xticks(range(len(perf_df)))
        ax.set_xticklabels(perf_df['norm_type'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, row in perf_df.iterrows():
            ax.text(i, row['mean_loss'] + row['std_loss'] + 0.01,
                   f'{row["mean_loss"]:.3f}\nn={row["count"]}',
                   ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('6b_normalization_performance_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved normalization performance comparison to: 6b_normalization_performance_comparison.png")
    # plt.show()  # Commented out for faster execution
    plt.close()
    
    # Print summary of normalization types found
    print("\n📋 Normalization Types Summary:")
    for norm_type in unique_norm_types:
        norm_data = valid_df[valid_df['norm_layer'] == norm_type]
        print(f"  {norm_type}: {len(norm_data)} runs")
        
        # Show performance stats
        final_losses = norm_data['final_train_loss'].dropna()
        if not final_losses.empty:
            print(f"    Performance: {final_losses.mean():.4f} ± {final_losses.std():.4f}")
        
        # Show model size distribution
        model_size_dist = norm_data['total_params'].value_counts().head(3)
        if not model_size_dist.empty:
            print(f"    Common model sizes: {dict(model_size_dist)}")
    print()


def create_data_summary_table(df, training_curves_df):
    """Create a comprehensive summary table of the analyzed data (DETERMINISTIC)"""
    print("\n" + "="*80)
    print("DETERMINISTIC TRAINING CURVES ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n--- Normalization Layer Analysis (ALL DATA - DETERMINISTIC) ---")
    print("Normalization Layer Types:")
    for norm_type in sorted(df['norm_layer'].dropna().unique()):
        norm_data = df[df['norm_layer'] == norm_type]
        count = len(norm_data)
        mean_final = norm_data['final_train_loss'].dropna().mean()
        
        print(f"  {norm_type}: {count} runs, Final loss: {mean_final:.4f}")
        
        # Quick training curve stats for this norm type
        norm_run_ids = norm_data['run_id'].tolist()
        norm_curves = training_curves_df[training_curves_df['run_id'].isin(norm_run_ids)]
        if not norm_curves.empty:
            print(f"    ALL data points: {len(norm_curves)}")
    
    print("\n--- Quantization Analysis (ALL DATA - DETERMINISTIC) ---")
    print("Bit Width Categories (excluding fixed 8-bit layers):")
    for bw in sorted(df['uniform_bit_width'].dropna().unique()):
        bw_data = df[df['uniform_bit_width'] == bw]
        count = len(bw_data)
        mean_final = bw_data['final_train_loss'].dropna().mean()
        
        print(f"  {int(bw)}-bit: {count} runs, Final loss: {mean_final:.4f}")
        
        # Quick training curve stats for this bit width
        bw_run_ids = bw_data['run_id'].tolist()
        bw_curves = training_curves_df[training_curves_df['run_id'].isin(bw_run_ids)]
        if not bw_curves.empty:
            print(f"    ALL data points: {len(bw_curves)}")

def main(loss_type='train'):
    """Main analysis function with training curves focus - DETERMINISTIC & REPRODUCIBLE"""
    # Map loss type to actual metric name in database
    loss_metric_map = {
        'train': 'train/loss',
        'val': 'validate/loss'
    }
    
    actual_metric = loss_metric_map[loss_type]
    print(f"🔍 Creating comprehensive training curves analysis using {loss_type} loss ({actual_metric})...")
    print("📈 Focus: Training loss progression over time instead of just final values")
    print("🎯 DETERMINISTIC: No random sampling - same results every run")
    print("Note: Excluding fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width analysis")
    
    conn = sqlite3.connect('wandb_data.db')
    
    # Extract detailed configuration data
    print("\n1. Extracting detailed configuration data...")
    df = extract_detailed_config_data(conn)
    print(df.head())
    
    # Get training curves for more runs
    print(f"2. Extracting training curves using {loss_type} loss ({actual_metric})...")
    # Use different min_steps for validation vs training since validation is logged less frequently
    min_steps = 5 if loss_type == 'val' else 100
    training_curves_df = get_training_curves(conn, max_steps=400, min_steps=min_steps, loss_type=actual_metric)
    print(f"   Found {len(training_curves_df)} training data points across ALL filtered runs")
    
    # Create comprehensive data summary
    create_data_summary_table(df, training_curves_df)
    
    # Create individual plots one by one for better visibility
    print("\n1a. Creating Plot 1a: Training Curves by Model Size (Subplots, Color = Bit Width)...")
    #plot_training_curves_by_model_size_subplots(df, training_curves_df)
    #
    #print("\n1b. Creating Plot 1b: Training Curves by Bit Width (Subplots, Color = Model Size)...")
    #plot_training_curves_by_bit_width_subplots(df, training_curves_df)
    #
    #print("\n1c. Creating Plot 1c: Training Curves Grouped by Model Size (Single Plot)...")
    #plot_training_curves_by_model_size(df, training_curves_df)
    #
    #print("\n1d. Creating Plot 1d: Training Curves Grouped by Bit Width (Single Plot)...")
    #plot_training_curves_by_bit_width(df, training_curves_df)
    #
    #print("\n2. Creating Plot 2: Average Convergence Curves...")
    #plot_average_convergence_curves(df, training_curves_df)
    #
    #print("\n3. Creating Plot 3: Convergence Performance Summary...")
    #plot_convergence_performance_summary(df)
    #
    #print("\n4. Creating Plot 4: Hyperparameter Analysis...")
    #plot_hyperparameter_curves_analysis(df, training_curves_df)

    #print("\n5. Creating Plot 5: Training Efficiency Analysis...")
    #plot_training_efficiency_analysis(df, training_curves_df)

    print(f"\n6. Creating Plot 6: Compare Normalization Layer Types (using {loss_type} loss)...")
    plot_compare_norm_layers(df, training_curves_df)
    
    print(f"\n6a-c. Creating Detailed Normalization Analysis by Hyperparameters (using {loss_type} loss)...")
    plot_detailed_norm_layers_by_hyperparams(df, training_curves_df, loss_type=loss_type)
    
    print(f"\n8. Creating Batch Size Comparison by Normalization Type (using {loss_type} loss)...")
    plot_batch_size_comparison_by_norm_layers(df, training_curves_df, loss_type=loss_type)
    
    print(f"\n8. Creating Batch Size Comparison Across Normalization Types (using {loss_type} loss)...")
    plot_batch_size_comparison_by_norm_layers(df, training_curves_df, loss_type=loss_type)

    # Save all data for further analysis
    df.to_csv(f'detailed_training_analysis_data_{loss_type}.csv', index=False)
    training_curves_df.to_csv(f'all_training_curves_data_{loss_type}.csv', index=False)
    print(f"\n💾 Saved detailed analysis data to: detailed_training_analysis_data_{loss_type}.csv")
    print(f"💾 Saved all training curves to: all_training_curves_data_{loss_type}.csv")
    
    conn.close()
    print(f"\n✅ Comprehensive training curves analysis complete using {loss_type} loss!")
    print("🎯 DETERMINISTIC: Same results guaranteed on every run (no random sampling)")
    print(f"\n📊 Generated individual plot files for {loss_type} loss:")
    print("  - 1a_training_curves_by_model_size_subplots.png (subplots by model size, color = bit width)")
    print("  - 1b_training_curves_by_bit_width_subplots.png (subplots by bit width, color = model size)")
    print("  - 1c_training_curves_by_model_size.png (single plot grouped by model size)")
    print("  - 1d_training_curves_by_bit_width.png (single plot grouped by bit width)")
    print("  - 2_average_convergence_curves.png")
    print("  - 3_convergence_performance_summary.png")
    print("  - 4_hyperparameter_analysis.png (colored by model size)")
    print("  - 5_training_efficiency_analysis.png")
    print("  - 6_normalization_layer_analysis.png (training curves by normalization type - separate subplots)")
    print("  - 6b_normalization_performance_comparison.png (performance comparison bar chart)")
    print(f"  - 7a_[norm]_batch[size]_{loss_type}.png (individual plots for each norm type + batch size combination)")
    print(f"  - 7b_[norm]_dropout[rate]_{loss_type}.png (individual plots for each norm type + dropout combination)")
    print(f"  - 7c_[norm]_weightdecay[value]_{loss_type}.png (individual plots for each norm type + weight decay combination)")
    print(f"  - 8_batch_size_comparison_by_norm_layers_{loss_type}.png (batch size comparison across normalization types)")
    print(f"  - 8b_batch_size_performance_summary_{loss_type}.png (batch size performance summary bar chart)")
    print("    Note: 'BatchNormIdPure' displayed as 'BatchNorm' for clarity, all plots use consistent y-axis scaling")
    print("    Enhancement: All plots now include both model size AND bit-width information")
    print("    Color coding: Model sizes use distinct colors (Set1 colormap)")
    print("    Line styles: 2-bit = dotted (:), 3-bit = dashed (--), 4-bit = solid (-)")
    print("    Batch size plots: Color = batch size, Line style = bit width")
    print("    Consistent y-axis scaling across all related plots for easy comparison")
    print("\n📊 Generated data files:")
    print(f"  - detailed_training_analysis_data_{loss_type}.csv")
    print(f"  - all_training_curves_data_{loss_type}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate detailed training analysis plots')
    parser.add_argument('--loss-type', type=str, choices=['train', 'val'], default='train',
                       help='Loss type to analyze: train for training loss, val for validation loss (default: train)')
    
    args = parser.parse_args()
    main(loss_type=args.loss_type)
