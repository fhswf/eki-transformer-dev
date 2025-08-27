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

def get_training_curves(conn, max_steps=400, min_steps=100):
    """Extract training curves with all steps, filtering out short runs - DETERMINISTIC"""
    # Get ALL runs deterministically (no random sampling)
    all_runs_query = f"""
        SELECT DISTINCT run_id 
        FROM metrics 
        WHERE metric_name = 'train/loss'
        ORDER BY run_id
    """
    all_runs_df = pd.read_sql_query(all_runs_query, conn)
    run_ids = "', '".join(all_runs_df['run_id'].tolist())
    
    # Get ALL steps for ALL runs (completely deterministic)
    training_curves_query = f"""
        SELECT run_id, step, metric_value
        FROM metrics
        WHERE metric_name = 'train/loss'
        AND run_id IN ('{run_ids}')
        AND step <= {max_steps}
        ORDER BY run_id, step
    """
    print(f"   Getting ALL steps for ALL {len(all_runs_df)} runs (max {max_steps} steps)...")
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


def create_data_summary_table(df, training_curves_df):
    """Create a comprehensive summary table of the analyzed data (DETERMINISTIC)"""
    print("\n" + "="*80)
    print("DETERMINISTIC TRAINING CURVES ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n--- Quantization Analysis (ALL DATA - DETERMINISTIC) ---")
    print("Bit Width Categories (excluding fixed 8-bit layers):")
    for bw in sorted(df['uniform_bit_width'].dropna().unique()):
        bw_data = df[df['uniform_bit_width'] == bw]
        count = len(bw_data)
        mean_final = bw_data['final_train_loss'].mean()
        
        print(f"  {int(bw)}-bit: {count} runs, Final loss: {mean_final:.4f}")
        
        # Quick training curve stats for this bit width
        bw_run_ids = bw_data['run_id'].tolist()
        bw_curves = training_curves_df[training_curves_df['run_id'].isin(bw_run_ids)]
        if not bw_curves.empty:
            print(f"    ALL data points: {len(bw_curves)}")

def main():
    """Main analysis function with training curves focus - DETERMINISTIC & REPRODUCIBLE"""
    print("🔍 Creating comprehensive training curves analysis...")
    print("📈 Focus: Training loss progression over time instead of just final values")
    print("🎯 DETERMINISTIC: No random sampling - same results every run")
    print("Note: Excluding fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width analysis")
    
    conn = sqlite3.connect('wandb_data.db')
    
    # Extract detailed configuration data
    print("\n1. Extracting detailed configuration data...")
    df = extract_detailed_config_data(conn)
    print(df.head())
    
    # Get training curves for more runs
    print("2. Extracting training curves...")
    training_curves_df = get_training_curves(conn, max_steps=400, min_steps=100)
    print(f"   Found {len(training_curves_df)} training data points across ALL filtered runs")
    
    # Create comprehensive data summary
    create_data_summary_table(df, training_curves_df)
    
    # Create individual plots one by one for better visibility
    print("\n3a. Creating Plot 1a: Training Curves by Model Size (Subplots, Color = Bit Width)...")
    plot_training_curves_by_model_size_subplots(df, training_curves_df)
    
    print("\n3b. Creating Plot 1b: Training Curves by Bit Width (Subplots, Color = Model Size)...")
    plot_training_curves_by_bit_width_subplots(df, training_curves_df)
    
    print("\n3c. Creating Plot 1c: Training Curves Grouped by Model Size (Single Plot)...")
    plot_training_curves_by_model_size(df, training_curves_df)
    
    print("\n3d. Creating Plot 1d: Training Curves Grouped by Bit Width (Single Plot)...")
    plot_training_curves_by_bit_width(df, training_curves_df)
    
    print("\n4. Creating Plot 2: Average Convergence Curves...")
    plot_average_convergence_curves(df, training_curves_df)
    
    print("\n5. Creating Plot 3: Convergence Performance Summary...")
    plot_convergence_performance_summary(df)
    
    print("\n6. Creating Plot 4: Hyperparameter Analysis...")
    plot_hyperparameter_curves_analysis(df, training_curves_df)

    print("\n7. Creating Plot 5: Training Efficiency Analysis...")
    plot_training_efficiency_analysis(df, training_curves_df)
    
    # Save all data for further analysis
    df.to_csv('detailed_training_analysis_data.csv', index=False)
    training_curves_df.to_csv('all_training_curves_data.csv', index=False)
    print("\n💾 Saved detailed analysis data to: detailed_training_analysis_data.csv")
    print("💾 Saved all training curves to: all_training_curves_data.csv")
    
    conn.close()
    print("\n✅ Comprehensive training curves analysis complete!")
    print("🎯 DETERMINISTIC: Same results guaranteed on every run (no random sampling)")
    print("\n📊 Generated individual plot files:")
    print("  - 1a_training_curves_by_model_size_subplots.png (subplots by model size, color = bit width)")
    print("  - 1b_training_curves_by_bit_width_subplots.png (subplots by bit width, color = model size)")
    print("  - 1c_training_curves_by_model_size.png (single plot grouped by model size)")
    print("  - 1d_training_curves_by_bit_width.png (single plot grouped by bit width)")
    print("  - 2_average_convergence_curves.png")
    print("  - 3_convergence_performance_summary.png")
    print("  - 4_hyperparameter_analysis.png (colored by model size)")
    print("  - 5_training_efficiency_analysis.png")
    print("\n📊 Generated data files:")
    print("  - detailed_training_analysis_data.csv")
    print("  - all_training_curves_data.csv")

if __name__ == "__main__":
    main()
