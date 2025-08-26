"""
Demo script showing how to use your extracted wandb data
with focus on model size and quantization parameter comparisons
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def demo_analysis():
    """Demonstrate analysis of the extracted wandb data with model and quantization comparisons"""
    conn = sqlite3.connect('wandb_data.db')
    
    print("🔍 Analyzing your extracted wandb data...")
    
    # 1. Show available runs
    print("\n=== Available Runs ===")
    runs = pd.read_sql_query("""
        SELECT run_id, name, state, created_at 
        FROM runs 
        ORDER BY created_at DESC
    """, conn)
    print(runs)
    
    # 2. Extract and analyze model configurations
    print("\n=== Model Configuration Analysis ===")
    config_analysis = analyze_model_configs(conn)
    print(config_analysis)
    
    # 3. Create model size comparison plots
    print("\n=== Creating Model Size Comparison Plots ===")
    plot_model_size_comparison(conn)
    
    # 4. Create quantization parameter comparison plots
    print("\n=== Creating Quantization Parameter Comparison Plots ===")
    plot_quantization_comparison(conn)
    
    # 5. Performance vs Model Size Analysis
    print("\n=== Performance vs Model Size Analysis ===")
    plot_performance_vs_model_size(conn)
    
    # 6. Performance vs Quantization Analysis
    print("\n=== Performance vs Quantization Analysis ===")
    plot_performance_vs_quantization(conn)
    
    # 7. Training curves comparison grouped by model properties
    print("\n=== Training Curves by Model Properties ===")
    plot_training_curves_by_properties(conn)
    
    conn.close()
    print("\n✅ Analysis complete!")

def extract_config_data(conn):
    """Extract and parse configuration data from all runs"""
    runs_df = pd.read_sql_query("""
        SELECT run_id, name, config, summary, state
        FROM runs
    """, conn)
    
    config_data = []
    for _, row in runs_df.iterrows():
        try:
            config = json.loads(row['config'])
            summary = json.loads(row['summary'])
            
            # Extract model parameters
            model_config = config.get('model', {}).get('args', {})
            quant_config = config.get('quantization', {})
            
            # Calculate model size metrics
            n_embd = model_config.get('n_embd', 0)
            n_layer = model_config.get('n_layer', 0)
            n_head = model_config.get('n_head', 0)
            vocab_size = model_config.get('vocab_size', 0)
            block_size = model_config.get('block_size', 0)
            
            # Estimate model parameters (rough calculation)
            # Transformer parameter estimation: embedding + layers + output
            embed_params = vocab_size * n_embd
            layer_params = n_layer * (4 * n_embd * n_embd + 2 * n_embd)  # Attention + MLP + LayerNorm
            total_params = embed_params + layer_params
            
            # Extract quantization info
            quant_enabled = quant_config.get('quantize', False)
            quant_type = quant_config.get('type', 'None')
            
            # Extract bit widths from quantization layers
            bit_widths = []
            if 'model' in quant_config and 'layers' in quant_config['model']:
                for layer_name, layer_config in quant_config['model']['layers'].items():
                    if 'quantizers' in layer_config:
                        for quant_name, quant_config_item in layer_config['quantizers'].items():
                            if isinstance(quant_config_item, dict) and 'args' in quant_config_item:
                                bit_width = quant_config_item['args'].get('bit_width')
                                if bit_width:
                                    bit_widths.append(bit_width)
            
            avg_bit_width = np.mean(bit_widths) if bit_widths else None
            min_bit_width = min(bit_widths) if bit_widths else None
            max_bit_width = max(bit_widths) if bit_widths else None
            
            data_row = {
                'run_id': row['run_id'],
                'name': row['name'],
                'state': row['state'],
                'n_embd': n_embd,
                'n_layer': n_layer,
                'n_head': n_head,
                'vocab_size': vocab_size,
                'block_size': block_size,
                'total_params_est': total_params,
                'quant_enabled': quant_enabled,
                'quant_type': quant_type,
                'avg_bit_width': avg_bit_width,
                'min_bit_width': min_bit_width,
                'max_bit_width': max_bit_width,
                'num_quant_layers': len(bit_widths),
                'final_train_loss': summary.get('train/loss'),
                'final_val_loss': summary.get('validate/loss'),
                'learning_rate': config.get('optim', {}).get('args', {}).get('learning_rate'),
                'batch_size': config.get('dataset', {}).get('dataloader', {}).get('batch_size'),
            }
            config_data.append(data_row)
            
        except Exception as e:
            print(f"Error parsing config for run {row['run_id']}: {e}")
    
    return pd.DataFrame(config_data)

def analyze_model_configs(conn):
    """Analyze model configurations"""
    config_df = extract_config_data(conn)
    
    print("\n--- Model Architecture Summary ---")
    print(f"Model embedding dimensions: {config_df['n_embd'].unique()}")
    print(f"Number of layers: {config_df['n_layer'].unique()}")
    print(f"Number of attention heads: {config_df['n_head'].unique()}")
    print(f"Estimated total parameters: {config_df['total_params_est'].unique()}")
    
    print("\n--- Quantization Summary ---")
    print(f"Quantization enabled: {config_df['quant_enabled'].unique()}")
    print(f"Quantization types: {config_df['quant_type'].unique()}")
    print(f"Average bit widths: {config_df['avg_bit_width'].unique()}")
    print(f"Min bit widths: {config_df['min_bit_width'].unique()}")
    print(f"Max bit widths: {config_df['max_bit_width'].unique()}")
    
    return config_df

def plot_model_size_comparison(conn):
    """Create plots comparing model sizes"""
    config_df = extract_config_data(conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Architecture Comparison', fontsize=16)
    
    # 1. Model size distribution
    axes[0, 0].hist(config_df['total_params_est'], bins=10, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Estimated Total Parameters')
    axes[0, 0].set_ylabel('Number of Runs')
    axes[0, 0].set_title('Model Size Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Embedding dimension vs Layers
    scatter = axes[0, 1].scatter(config_df['n_embd'], config_df['n_layer'], 
                                c=config_df['total_params_est'], cmap='viridis', s=100)
    axes[0, 1].set_xlabel('Embedding Dimension')
    axes[0, 1].set_ylabel('Number of Layers')
    axes[0, 1].set_title('Model Architecture (colored by total params)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Total Parameters')
    
    # 3. Model size vs Performance (if available)
    valid_loss_data = config_df.dropna(subset=['final_train_loss'])
    if not valid_loss_data.empty:
        axes[1, 0].scatter(valid_loss_data['total_params_est'], valid_loss_data['final_train_loss'], 
                          s=100, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Estimated Total Parameters')
        axes[1, 0].set_ylabel('Final Training Loss')
        axes[1, 0].set_title('Model Size vs Training Performance')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No training loss data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Model Size vs Training Performance (No Data)')
    
    # 4. Architecture configuration heatmap
    arch_cols = ['n_embd', 'n_layer', 'n_head', 'vocab_size']
    arch_data = config_df[arch_cols].T
    im = axes[1, 1].imshow(arch_data.values, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_xticks(range(len(config_df)))
    axes[1, 1].set_xticklabels([name[:15] + '...' for name in config_df['name']], rotation=45)
    axes[1, 1].set_yticks(range(len(arch_cols)))
    axes[1, 1].set_yticklabels(arch_cols)
    axes[1, 1].set_title('Architecture Parameters Heatmap')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('model_size_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved model size comparison to: model_size_comparison.png")
    plt.show()

def plot_quantization_comparison(conn):
    """Create plots comparing quantization parameters"""
    config_df = extract_config_data(conn)
    
    # Filter for quantized models
    quant_df = config_df[config_df['quant_enabled'] == True]
    
    if quant_df.empty:
        print("No quantized models found in the data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantization Parameter Comparison', fontsize=16)
    
    # 1. Bit width distribution
    all_bit_widths = []
    for _, row in quant_df.iterrows():
        if pd.notna(row['avg_bit_width']):
            all_bit_widths.append(row['avg_bit_width'])
    
    if all_bit_widths:
        axes[0, 0].hist(all_bit_widths, bins=10, alpha=0.7, color='lightcoral')
        axes[0, 0].set_xlabel('Average Bit Width')
        axes[0, 0].set_ylabel('Number of Runs')
        axes[0, 0].set_title('Quantization Bit Width Distribution')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Quantization impact on model size
    axes[0, 1].scatter(quant_df['total_params_est'], quant_df['avg_bit_width'], 
                      s=100, alpha=0.7, color='purple')
    axes[0, 1].set_xlabel('Model Size (Parameters)')
    axes[0, 1].set_ylabel('Average Bit Width')
    axes[0, 1].set_title('Model Size vs Quantization Level')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Quantization vs Performance
    valid_quant_loss = quant_df.dropna(subset=['final_train_loss', 'avg_bit_width'])
    if not valid_quant_loss.empty:
        axes[1, 0].scatter(valid_quant_loss['avg_bit_width'], valid_quant_loss['final_train_loss'],
                          s=100, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Average Bit Width')
        axes[1, 0].set_ylabel('Final Training Loss')
        axes[1, 0].set_title('Quantization Level vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Quantization parameters summary
    quant_summary = quant_df[['name', 'avg_bit_width', 'min_bit_width', 'max_bit_width', 'num_quant_layers']].fillna(0)
    quant_cols = ['avg_bit_width', 'min_bit_width', 'max_bit_width', 'num_quant_layers']
    
    if not quant_summary.empty:
        im = axes[1, 1].imshow(quant_summary[quant_cols].T.values, cmap='RdYlBu_r', aspect='auto')
        axes[1, 1].set_xticks(range(len(quant_summary)))
        axes[1, 1].set_xticklabels([name[:15] + '...' for name in quant_summary['name']], rotation=45)
        axes[1, 1].set_yticks(range(len(quant_cols)))
        axes[1, 1].set_yticklabels(quant_cols)
        axes[1, 1].set_title('Quantization Parameters Heatmap')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('quantization_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved quantization comparison to: quantization_comparison.png")
    plt.show()

def plot_performance_vs_model_size(conn):
    """Plot training performance vs model size"""
    config_df = extract_config_data(conn)
    
    # Get training curves
    loss_data = pd.read_sql_query("""
        SELECT r.name, r.run_id, m.step, m.metric_value
        FROM metrics m
        JOIN runs r ON m.run_id = r.run_id
        WHERE m.metric_name = 'train/loss'
        ORDER BY r.name, m.step
    """, conn)
    
    if loss_data.empty:
        print("No training loss data available for performance comparison")
        return
    
    # Merge with config data
    merged_data = loss_data.merge(config_df[['run_id', 'total_params_est', 'name']], on='run_id')
    
    plt.figure(figsize=(14, 8))
    
    # Create subplot for training curves colored by model size
    plt.subplot(1, 2, 1)
    unique_runs = merged_data['run_id'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_runs)))
    
    for i, run_id in enumerate(unique_runs):
        run_data = merged_data[merged_data['run_id'] == run_id]
        model_size = run_data['total_params_est'].iloc[0]
        plt.plot(run_data['step'], run_data['metric_value'], 
                color=colors[i], label=f'{model_size/1e6:.1f}M params', linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curves by Model Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Final loss vs model size
    plt.subplot(1, 2, 2)
    final_losses = []
    model_sizes = []
    
    for run_id in unique_runs:
        run_data = merged_data[merged_data['run_id'] == run_id]
        final_loss = run_data['metric_value'].iloc[-1]  # Last recorded loss
        model_size = run_data['total_params_est'].iloc[0]
        final_losses.append(final_loss)
        model_sizes.append(model_size)
    
    plt.scatter(model_sizes, final_losses, s=100, alpha=0.7, color='red')
    plt.xlabel('Model Size (Parameters)')
    plt.ylabel('Final Training Loss')
    plt.title('Final Loss vs Model Size')
    plt.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, (size, loss) in enumerate(zip(model_sizes, final_losses)):
        plt.annotate(f'{size/1e6:.1f}M', (size, loss), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_vs_model_size.png', dpi=150, bbox_inches='tight')
    print("📊 Saved performance vs model size to: performance_vs_model_size.png")
    plt.show()

def plot_performance_vs_quantization(conn):
    """Plot training performance vs quantization parameters"""
    config_df = extract_config_data(conn)
    quant_df = config_df[config_df['quant_enabled'] == True]
    
    if quant_df.empty:
        print("No quantized models found for quantization comparison")
        return
    
    # Get training curves for quantized models
    quant_run_ids = "', '".join(quant_df['run_id'].tolist())
    loss_data = pd.read_sql_query(f"""
        SELECT r.name, r.run_id, m.step, m.metric_value
        FROM metrics m
        JOIN runs r ON m.run_id = r.run_id
        WHERE m.metric_name = 'train/loss' AND r.run_id IN ('{quant_run_ids}')
        ORDER BY r.name, m.step
    """, conn)
    
    if loss_data.empty:
        print("No training loss data available for quantized models")
        return
    
    # Merge with quantization config data
    merged_data = loss_data.merge(
        quant_df[['run_id', 'avg_bit_width', 'min_bit_width', 'name']], 
        on='run_id'
    )
    
    plt.figure(figsize=(14, 8))
    
    # Training curves colored by bit width
    plt.subplot(1, 2, 1)
    unique_runs = merged_data['run_id'].unique()
    
    for run_id in unique_runs:
        run_data = merged_data[merged_data['run_id'] == run_id]
        avg_bit_width = run_data['avg_bit_width'].iloc[0]
        if pd.notna(avg_bit_width):
            plt.plot(run_data['step'], run_data['metric_value'], 
                    label=f'{avg_bit_width:.1f} bit avg', linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curves by Quantization Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Final loss vs bit width
    plt.subplot(1, 2, 2)
    final_losses = []
    bit_widths = []
    
    for run_id in unique_runs:
        run_data = merged_data[merged_data['run_id'] == run_id]
        final_loss = run_data['metric_value'].iloc[-1]
        avg_bit_width = run_data['avg_bit_width'].iloc[0]
        if pd.notna(avg_bit_width):
            final_losses.append(final_loss)
            bit_widths.append(avg_bit_width)
    
    if final_losses and bit_widths:
        plt.scatter(bit_widths, final_losses, s=100, alpha=0.7, color='blue')
        plt.xlabel('Average Bit Width')
        plt.ylabel('Final Training Loss')
        plt.title('Final Loss vs Quantization Level')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(bit_widths) > 1:
            z = np.polyfit(bit_widths, final_losses, 1)
            p = np.poly1d(z)
            plt.plot(sorted(bit_widths), p(sorted(bit_widths)), "--", alpha=0.7, color='red')
    
    plt.tight_layout()
    plt.savefig('performance_vs_quantization.png', dpi=150, bbox_inches='tight')
    print("📊 Saved performance vs quantization to: performance_vs_quantization.png")
    plt.show()

def plot_training_curves_by_properties(conn):
    """Plot training curves grouped by model properties"""
    config_df = extract_config_data(conn)
    
    # Get all training loss data
    loss_data = pd.read_sql_query("""
        SELECT r.name, r.run_id, m.step, m.metric_value
        FROM metrics m
        JOIN runs r ON m.run_id = r.run_id
        WHERE m.metric_name = 'train/loss'
        ORDER BY r.name, m.step
    """, conn)
    
    if loss_data.empty:
        print("No training loss data available")
        return
    
    # Merge with config data
    merged_data = loss_data.merge(config_df[['run_id', 'name', 'n_layer', 'n_embd', 'quant_enabled']], on='run_id')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves Grouped by Model Properties', fontsize=16)
    
    # 1. Group by number of layers
    axes[0, 0].set_title('Training Loss by Number of Layers')
    for n_layer in merged_data['n_layer'].unique():
        layer_data = merged_data[merged_data['n_layer'] == n_layer]
        for run_id in layer_data['run_id'].unique():
            run_data = layer_data[layer_data['run_id'] == run_id]
            axes[0, 0].plot(run_data['step'], run_data['metric_value'], 
                           label=f'{n_layer} layers', alpha=0.7)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Group by embedding dimension
    axes[0, 1].set_title('Training Loss by Embedding Dimension')
    for n_embd in merged_data['n_embd'].unique():
        embd_data = merged_data[merged_data['n_embd'] == n_embd]
        for run_id in embd_data['run_id'].unique():
            run_data = embd_data[embd_data['run_id'] == run_id]
            axes[0, 1].plot(run_data['step'], run_data['metric_value'], 
                           label=f'{n_embd} embd', alpha=0.7)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Group by quantization status
    axes[1, 0].set_title('Training Loss by Quantization Status')
    for quant_enabled in merged_data['quant_enabled'].unique():
        quant_data = merged_data[merged_data['quant_enabled'] == quant_enabled]
        for run_id in quant_data['run_id'].unique():
            run_data = quant_data[quant_data['run_id'] == run_id]
            label = 'Quantized' if quant_enabled else 'Full Precision'
            axes[1, 0].plot(run_data['step'], run_data['metric_value'], 
                           label=label, alpha=0.7)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. All runs with different colors
    axes[1, 1].set_title('All Training Curves')
    colors = plt.cm.Set3(np.linspace(0, 1, len(merged_data['run_id'].unique())))
    for i, run_id in enumerate(merged_data['run_id'].unique()):
        run_data = merged_data[merged_data['run_id'] == run_id]
        # Use run_id as label since we have limited columns
        axes[1, 1].plot(run_data['step'], run_data['metric_value'], 
                       color=colors[i], label=f'Run {i+1}', alpha=0.8)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_curves_by_properties.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training curves by properties to: training_curves_by_properties.png")
    plt.show()

if __name__ == "__main__":
    demo_analysis()

if __name__ == "__main__":
    demo_analysis()
