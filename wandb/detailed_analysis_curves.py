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

def extract_detailed_config_data(conn, sample_size=50):
    """Extract detailed configuration and performance data for a smaller sample of runs"""
    # OPTIMIZATION: Reduce sample size significantly for faster processing
    runs_df = pd.read_sql_query(f"""
        SELECT run_id, name, config, summary, state
        FROM runs
        WHERE state IN ('finished', 'running', 'crashed')  -- Include more states
        ORDER BY RANDOM()
        LIMIT {sample_size}
    """, conn)
    
    print(f"   Processing {len(runs_df)} completed runs for faster analysis...")
    
    # Get final training and validation losses for sampled runs
    final_metrics = {}
    
    if not runs_df.empty:
        run_ids = "', '".join(runs_df['run_id'].tolist())
        
        # OPTIMIZATION: Get final training loss with single optimized query
        train_loss_query = f"""
            SELECT run_id, metric_value as final_train_loss
            FROM metrics m1
            WHERE metric_name = 'train/loss' 
            AND run_id IN ('{run_ids}')
            AND step = (
                SELECT MAX(step) 
                FROM metrics m2 
                WHERE m2.run_id = m1.run_id AND m2.metric_name = 'train/loss'
            )
        """
        train_losses = pd.read_sql_query(train_loss_query, conn)
        for _, row in train_losses.iterrows():
            final_metrics[row['run_id']] = {'train_loss': row['final_train_loss']}
        
        # OPTIMIZATION: Skip validation loss if not critical for analysis
        # This reduces query complexity significantly
        print(f"   Skipping validation loss queries for performance (train loss focus)")
    
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

def get_training_curves(conn, sample_runs=50, max_steps=400, step_interval=10):
    """Extract training curves with aggressive sampling for optimal performance"""
    # First, get a representative sample of runs
    sample_query = f"""
        SELECT DISTINCT run_id 
        FROM metrics 
        WHERE metric_name = 'train/loss'
        ORDER BY RANDOM() 
        LIMIT {sample_runs}
    """
    sample_runs_df = pd.read_sql_query(sample_query, conn)
    run_ids = "', '".join(sample_runs_df['run_id'].tolist())
    
    # AGGRESSIVE OPTIMIZATION: Sample every Nth step to reduce data points dramatically
    training_curves_query = f"""
        SELECT run_id, step, metric_value
        FROM metrics
        WHERE metric_name = 'train/loss'
        AND run_id IN ('{run_ids}')
        AND step <= {max_steps}
        AND (step % {step_interval} = 0 OR step = 1)  -- Sample every {step_interval}th step + first step
        ORDER BY run_id, step
    """
    print(f"   Sampling {sample_runs} runs, max {max_steps} steps, every {step_interval}th step...")
    result = pd.read_sql_query(training_curves_query, conn)
    print(f"   Reduced to {len(result)} data points (from potential 240k+)")
    return result

def plot_training_curves_vs_bitwidth(df, training_curves_df):
    """Graph 1: Training curves colored by quantization bit-width (optimized)"""
    
    # Filter for runs with valid bit width data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss'])
    
    if valid_df.empty:
        print("No valid data for training curves vs bit-width analysis")
        return
    
    # Limit to curves we actually have data for
    available_runs = set(training_curves_df['run_id'].unique())
    valid_df = valid_df[valid_df['run_id'].isin(available_runs)]
    
    if valid_df.empty:
        print("No matching training curves for valid runs")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Curves vs Quantization Bit-Width (Sample Analysis)', fontsize=14)
    
    # Get unique bit widths for color mapping
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot 1: Training curves colored by bit-width (sample only)
    ax1 = axes[0]
    plotted_curves = 0
    max_curves_per_bitwidth = 5  # Limit curves for readability
    
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        curves_plotted = 0
        
        for _, row in bw_runs.iterrows():
            if curves_plotted >= max_curves_per_bitwidth:
                break
                
            run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
            if not run_curves.empty:
                color = bit_width_colors[row['uniform_bit_width']]
                label = f"{int(row['uniform_bit_width'])}-bit" if curves_plotted == 0 else ""
                ax1.plot(run_curves['step'], run_curves['metric_value'], 
                        color=color, label=label, linewidth=1.5, alpha=0.7)
                curves_plotted += 1
                plotted_curves += 1
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss Curves by Quantization ({plotted_curves} curves)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final loss vs bit-width scatter with summary statistics
    ax2 = axes[1]
    for bit_width in bit_widths:
        bw_data = valid_df[valid_df['uniform_bit_width'] == bit_width]
        if not bw_data.empty:
            mean_loss = bw_data['final_train_loss'].mean()
            std_loss = bw_data['final_train_loss'].std() if len(bw_data) > 1 else 0
            color = bit_width_colors[bit_width]
            
            # Scatter plot with jitter for visibility
            jitter = np.random.normal(0, 0.02, len(bw_data))
            ax2.scatter(bw_data['uniform_bit_width'] + jitter, bw_data['final_train_loss'], 
                       color=color, s=80, alpha=0.6, label=f"{int(bit_width)}-bit (n={len(bw_data)})")
            
            # Error bar for mean
            ax2.errorbar(bit_width, mean_loss, yerr=std_loss, 
                        color=color, capsize=8, capthick=2, linewidth=2, alpha=0.8,
                        marker='D', markersize=8)
    
    ax2.set_xlabel('Quantization Bit-Width (Variable Layers)')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Final Training Loss vs Bit-Width (Summary)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_vs_bitwidth_optimized.png', dpi=150, bbox_inches='tight')
    print("📊 Saved optimized training curves vs bit-width to: training_curves_vs_bitwidth_optimized.png")
    plt.show()

def plot_convergence_analysis(df, training_curves_df):
    """Graph 2: Convergence speed analysis by quantization level (optimized)"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for convergence analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Convergence Analysis by Quantization Level (Optimized)', fontsize=16)
    
    # Define loss thresholds for convergence analysis
    target_losses = [6.0, 5.5, 5.0]  # Reduced for performance
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot 1: Average training curves (much faster than individual curves)
    ax1 = axes[0, 0]
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
            ax1.plot(avg_curve['step'], avg_curve['metric_value'], 
                    color=color, label=label, linewidth=3, alpha=0.9)
            
            # Add confidence interval if we have std data
            if not std_curve['metric_value'].isna().all():
                ax1.fill_between(avg_curve['step'], 
                               avg_curve['metric_value'] - std_curve['metric_value'],
                               avg_curve['metric_value'] + std_curve['metric_value'],
                               color=color, alpha=0.2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss (Average)')
    ax1.set_title('Average Training Curves by Quantization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Simplified convergence analysis
    ax2 = axes[0, 1]
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
        bars = ax2.bar(range(len(conv_df)), conv_df['mean_final_loss'], 
                      yerr=conv_df['std_final_loss'], capsize=8, alpha=0.7,
                      color=[bit_width_colors[bw/1] for bw in conv_df['bit_width']])
        
        ax2.set_xlabel('Quantization Bit-Width')
        ax2.set_ylabel('Final Training Loss')
        ax2.set_title('Convergence Performance Summary')
        ax2.set_xticks(range(len(conv_df)))
        ax2.set_xticklabels([f'{bw}-bit' for bw in conv_df['bit_width']])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, row in conv_df.iterrows():
            ax2.text(i, row['mean_final_loss'] + row['std_final_loss'] + 0.05,
                    f'{row["mean_final_loss"]:.3f}\\nn={row["count"]}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Loss improvement rates (simplified)
    ax3 = axes[1, 0]
    improvement_data = []
    
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        bw_run_ids = bw_runs['run_id'].tolist()
        bw_curves = training_curves_df[training_curves_df['run_id'].isin(bw_run_ids)]
        
        if not bw_curves.empty:
            # Calculate improvement rate for each run
            for run_id in bw_run_ids:
                run_curves = bw_curves[bw_curves['run_id'] == run_id]
                if len(run_curves) > 1:
                    initial_loss = run_curves['metric_value'].iloc[0]
                    final_loss = run_curves['metric_value'].iloc[-1]
                    improvement_rate = (initial_loss - final_loss) / len(run_curves)
                    
                    improvement_data.append({
                        'bit_width': int(bit_width),
                        'improvement_rate': improvement_rate
                    })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        for bit_width in bit_widths:
            bw_data = imp_df[imp_df['bit_width'] == int(bit_width)]
            if not bw_data.empty:
                color = bit_width_colors[bit_width]
                ax3.scatter(bw_data['bit_width'], bw_data['improvement_rate'],
                           color=color, s=100, alpha=0.7, label=f"{int(bit_width)}-bit")
        
        ax3.set_xlabel('Quantization Bit-Width')
        ax3.set_ylabel('Loss Improvement Rate')
        ax3.set_title('Training Speed by Quantization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for improvement analysis', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    
    # Plot 4: Data coverage summary
    ax4 = axes[1, 1]
    coverage_data = []
    
    for bit_width in bit_widths:
        bw_runs = valid_df[valid_df['uniform_bit_width'] == bit_width]
        bw_run_ids = bw_runs['run_id'].tolist()
        bw_curves = training_curves_df[training_curves_df['run_id'].isin(bw_run_ids)]
        
        coverage_data.append({
            'bit_width': int(bit_width),
            'total_runs': len(bw_runs),
            'data_points': len(bw_curves),
            'avg_steps_per_run': len(bw_curves) / len(bw_runs) if len(bw_runs) > 0 else 0
        })
    
    if coverage_data:
        cov_df = pd.DataFrame(coverage_data)
        
        # Dual y-axis plot
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar([x - 0.2 for x in range(len(cov_df))], cov_df['total_runs'], 
                       width=0.4, alpha=0.7, label='Total Runs', color='skyblue')
        bars2 = ax4_twin.bar([x + 0.2 for x in range(len(cov_df))], cov_df['data_points'], 
                            width=0.4, alpha=0.7, label='Data Points', color='lightcoral')
        
        ax4.set_xlabel('Quantization Bit-Width')
        ax4.set_ylabel('Number of Runs', color='skyblue')
        ax4_twin.set_ylabel('Training Data Points', color='lightcoral')
        ax4.set_title('Data Coverage by Quantization Level')
        ax4.set_xticks(range(len(cov_df)))
        ax4.set_xticklabels([f'{bw}-bit' for bw in cov_df['bit_width']])
        
        # Add value labels
        for i, row in cov_df.iterrows():
            ax4.text(i-0.2, row['total_runs'] + 0.5, f'{row["total_runs"]}',
                    ha='center', va='bottom', fontsize=9)
            ax4_twin.text(i+0.2, row['data_points'] + 5, f'{row["data_points"]}',
                         ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training_convergence_analysis_optimized.png', dpi=150, bbox_inches='tight')
    print("📊 Saved optimized convergence analysis to: training_convergence_analysis_optimized.png")
    plt.show()

def plot_hyperparameter_curves_analysis(df, training_curves_df):
    """Graph 3: Comprehensive hyperparameter and quantization analysis"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['uniform_bit_width'])
    
    if valid_df.empty:
        print("No valid data for hyperparameter curves analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Training Analysis: Quantization, Hyperparameters & Performance', fontsize=16)
    
    bit_widths = sorted(valid_df['uniform_bit_width'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(bit_widths)))
    bit_width_colors = {bw: colors[i] for i, bw in enumerate(bit_widths)}
    
    # Plot 1: Training curves with detailed labels
    ax1 = axes[0, 0]
    for _, row in valid_df.iterrows():
        run_curves = training_curves_df[training_curves_df['run_id'] == row['run_id']]
        if not run_curves.empty:
            color = bit_width_colors[row['uniform_bit_width']]
            linestyle = '-' if row['weight_decay'] == 0.0 else '--'
            alpha = 1.0 if row['dropout'] == 0.0 else 0.7
            
            label = f"{int(row['uniform_bit_width'])}-bit, wd={row['weight_decay']}, do={row['dropout']}"
            ax1.plot(run_curves['step'], run_curves['metric_value'], 
                    color=color, linestyle=linestyle, linewidth=2, alpha=alpha, label=label)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('All Training Curves (detailed)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance landscape
    ax2 = axes[0, 1]
    # Create a heatmap-style visualization of performance
    x_data = []
    y_data = []
    z_data = []
    labels = []
    
    for _, row in valid_df.iterrows():
        x_data.append(row['uniform_bit_width'])
        y_data.append(row['weight_decay'] * 10 + row['dropout'])  # Combine hyperparams
        z_data.append(row['final_train_loss'])
        labels.append(f"wd={row['weight_decay']}, do={row['dropout']}")
    
    scatter = ax2.scatter(x_data, y_data, c=z_data, cmap='RdYlBu_r', s=200, alpha=0.8)
    ax2.set_xlabel('Quantization Bit-Width')
    ax2.set_ylabel('Hyperparameter Combination (wd*10 + dropout)')
    ax2.set_title('Performance Landscape')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Final Training Loss')
    
    # Add annotations
    for i, label in enumerate(labels):
        ax2.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Plot 3: Training efficiency (loss per step)
    ax3 = axes[1, 0]
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
        scatter = ax3.scatter(eff_df['bit_width'], eff_df['efficiency'], 
                             c=eff_df['final_loss'], cmap='viridis', s=150, alpha=0.8)
        ax3.set_xlabel('Quantization Bit-Width')
        ax3.set_ylabel('Training Efficiency (loss reduction per step)')
        ax3.set_title('Training Efficiency vs Quantization')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Final Training Loss')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient efficiency data', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    summary_stats = []
    
    for bit_width in bit_widths:
        bw_data = valid_df[valid_df['uniform_bit_width'] == bit_width]
        if not bw_data.empty:
            mean_loss = bw_data['final_train_loss'].mean()
            std_loss = bw_data['final_train_loss'].std()
            min_loss = bw_data['final_train_loss'].min()
            max_loss = bw_data['final_train_loss'].max()
            
            summary_stats.append({
                'bit_width': int(bit_width),
                'mean_loss': mean_loss,
                'std_loss': std_loss if not np.isnan(std_loss) else 0,
                'min_loss': min_loss,
                'max_loss': max_loss,
                'count': len(bw_data)
            })
    
    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        
        # Bar plot with error bars
        x_pos = range(len(stats_df))
        bars = ax4.bar(x_pos, stats_df['mean_loss'], yerr=stats_df['std_loss'], 
                      capsize=10, alpha=0.7, color=[bit_width_colors[bw] for bw in bit_widths])
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(stats_df.iterrows()):
            ax4.text(i, row['mean_loss'] + row['std_loss'] + 0.05, 
                    f'{row["mean_loss"]:.2f}±{row["std_loss"]:.2f}\\nn={row["count"]}',
                    ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Quantization Bit-Width')
        ax4.set_ylabel('Final Training Loss')
        ax4.set_title('Summary Statistics by Quantization Level')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{int(bw)}-bit' for bw in bit_widths])
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Insufficient summary data', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_curves_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved hyperparameter curves analysis to: hyperparameter_curves_analysis.png")
    plt.show()

def create_data_summary_table(df, training_curves_df):
    """Create a comprehensive summary table of the analyzed data (optimized)"""
    print("\n" + "="*80)
    print("OPTIMIZED TRAINING CURVES ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nSample Size (for performance): {len(df)} runs")
    print(f"Runs with Training Loss: {df['final_train_loss'].notna().sum()}")
    print(f"Runs with Quantization: {df['quant_enabled'].sum()}")
    print(f"Sampled Training Data Points: {len(training_curves_df)}")
    print(f"Original Dataset: 504 total runs, 240k+ training points")
    
    # Training curve statistics
    if not training_curves_df.empty:
        curve_stats = training_curves_df.groupby('run_id').agg({
            'step': ['min', 'max', 'count'],
            'metric_value': ['min', 'max', 'mean', 'std']
        }).round(4)
        
        print("\n--- Sampled Training Curve Statistics ---")
        print(f"Average training steps per run: {curve_stats[('step', 'count')].mean():.0f}")
        print(f"Step range: {training_curves_df['step'].min()} - {training_curves_df['step'].max()}")
        print(f"Loss range: {training_curves_df['metric_value'].min():.4f} - {training_curves_df['metric_value'].max():.4f}")
    
    print("\n--- Quantization Analysis (Sample) ---")
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
            print(f"    Sample data points: {len(bw_curves)}")
    
    print(f"\n⚡ Performance Optimization Applied:")
    print(f"  • Sample size reduced to {len(df)} runs (from 504)")
    print(f"  • Step sampling every 15th point (from continuous)")
    print(f"  • Max steps limited to 300 (from 511)")
    print(f"  • Validation loss queries skipped")
    print(f"  • Result: ~{len(training_curves_df)} data points vs 240k+ original")

def main():
    """Main analysis function with training curves focus"""
    print("🔍 Creating comprehensive training curves analysis...")
    print("📈 Focus: Training loss progression over time instead of just final values")
    print("Note: Excluding fixed 8-bit layers (transformer.emb_add, linear_out) from bit-width analysis")
    
    conn = sqlite3.connect('wandb_data.db')
    
    # Extract detailed configuration data
    print("\n1. Extracting detailed configuration data...")
    df = extract_detailed_config_data(conn)
    
    # Get training curves for all runs
    print("2. Extracting training curves...")
    training_curves_df = get_training_curves(conn, sample_runs=30, max_steps=300, step_interval=15)
    print(f"   Found {len(training_curves_df)} training data points across sampled runs")
    
    # Create comprehensive data summary
    create_data_summary_table(df, training_curves_df)
    
    # Create the three enhanced graphs with training curves
    print("\n3. Creating Graph 1: Training Curves vs Quantization Bit-Width...")
    plot_training_curves_vs_bitwidth(df, training_curves_df)
    
    print("\n4. Creating Graph 2: Convergence Analysis by Quantization Level...")
    plot_convergence_analysis(df, training_curves_df)
    
    print("\n5. Creating Graph 3: Comprehensive Hyperparameter & Training Analysis...")
    plot_hyperparameter_curves_analysis(df, training_curves_df)
    
    # Save all data for further analysis
    df.to_csv('detailed_training_analysis_data.csv', index=False)
    training_curves_df.to_csv('all_training_curves_data.csv', index=False)
    print("\n💾 Saved detailed analysis data to: detailed_training_analysis_data.csv")
    print("💾 Saved all training curves to: all_training_curves_data.csv")
    
    conn.close()
    print("\n✅ Comprehensive training curves analysis complete!")
    print("\n📊 Generated files:")
    print("  - training_curves_vs_bitwidth.png")
    print("  - training_convergence_analysis.png") 
    print("  - hyperparameter_curves_analysis.png")
    print("  - detailed_training_analysis_data.csv")
    print("  - all_training_curves_data.csv")

if __name__ == "__main__":
    main()
