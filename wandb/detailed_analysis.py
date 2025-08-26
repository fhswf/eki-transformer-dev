"""
Detailed analysis script focusing on specific model size and quantization comparisons
as requested by the user.

Creates three specific graphs:
1. Grouped by model size: train/val loss vs quantization bit-width
2. Grouped by bit-width: train/val loss vs model size
3. Same as #2 but with hyperparameter separation
"""

import sqlite3
import pandas as pd
import matplotli    print("\n--- Quantization Variations ---")
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
            print(f"  {bw_cat}: {count} runs")as plt
import json
import numpy as np
from collections import defaultdict

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
                from collections import Counter
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

def plot_loss_vs_bitwidth_by_model_size(df):
    """Graph 1: Grouped by model size, plot train/val loss vs quantization bit-width"""
    
    # Filter for runs with valid bit width and loss data
    valid_df = df.dropna(subset=['uniform_bit_width', 'final_train_loss', 'final_val_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs bit-width analysis")
        return
    
    # Group by model size
    model_sizes = valid_df['model_size_category'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training and Validation Loss vs Quantization Bit-Width (Grouped by Model Size)', fontsize=14)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_sizes)))
    
    # Training Loss vs Bit Width
    for i, model_size in enumerate(model_sizes):
        size_data = valid_df[valid_df['model_size_category'] == model_size]
        axes[0].scatter(size_data['uniform_bit_width'], size_data['final_train_loss'], 
                       color=colors[i], label=model_size, s=100, alpha=0.7)
        
        # Add trend line if more than 1 point
        if len(size_data) > 1:
            z = np.polyfit(size_data['uniform_bit_width'], size_data['final_train_loss'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(size_data['uniform_bit_width'].min(), size_data['uniform_bit_width'].max(), 100)
            axes[0].plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.5)
    
    axes[0].set_xlabel('Quantization Bit-Width')
    axes[0].set_ylabel('Final Training Loss')
    axes[0].set_title('Training Loss vs Bit-Width')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation Loss vs Bit Width  
    for i, model_size in enumerate(model_sizes):
        size_data = valid_df[valid_df['model_size_category'] == model_size]
        axes[1].scatter(size_data['uniform_bit_width'], size_data['final_val_loss'], 
                       color=colors[i], label=model_size, s=100, alpha=0.7)
        
        # Add trend line if more than 1 point
        if len(size_data) > 1:
            z = np.polyfit(size_data['uniform_bit_width'], size_data['final_val_loss'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(size_data['uniform_bit_width'].min(), size_data['uniform_bit_width'].max(), 100)
            axes[1].plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Quantization Bit-Width')
    axes[1].set_ylabel('Final Validation Loss')
    axes[1].set_title('Validation Loss vs Bit-Width')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add data point annotations
    for i, row in valid_df.iterrows():
        axes[0].annotate(f'{row["uniform_bit_width"]:.1f}', 
                        (row['uniform_bit_width'], row['final_train_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1].annotate(f'{row["uniform_bit_width"]:.1f}', 
                        (row['uniform_bit_width'], row['final_val_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('loss_vs_bitwidth_by_model_size.png', dpi=150, bbox_inches='tight')
    print("📊 Saved loss vs bit-width (by model size) to: loss_vs_bitwidth_by_model_size.png")
    plt.show()

def plot_loss_vs_model_size_by_bitwidth(df):
    """Graph 2: Grouped by bit-width, plot train/val loss vs model size"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['total_params', 'final_train_loss', 'final_val_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs model size analysis")
        return
    
    # Group by bit width
    bit_widths = valid_df['bit_width_category'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training and Validation Loss vs Model Size (Grouped by Bit-Width)', fontsize=14)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(bit_widths)))
    
    # Training Loss vs Model Size
    for i, bit_width in enumerate(bit_widths):
        bw_data = valid_df[valid_df['bit_width_category'] == bit_width]
        axes[0].scatter(bw_data['total_params'], bw_data['final_train_loss'], 
                       color=colors[i], label=bit_width, s=100, alpha=0.7)
        
        # Add trend line if more than 1 point
        if len(bw_data) > 1:
            z = np.polyfit(bw_data['total_params'], bw_data['final_train_loss'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(bw_data['total_params'].min(), bw_data['total_params'].max(), 100)
            axes[0].plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.5)
    
    axes[0].set_xlabel('Model Size (Parameters)')
    axes[0].set_ylabel('Final Training Loss')
    axes[0].set_title('Training Loss vs Model Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation Loss vs Model Size
    for i, bit_width in enumerate(bit_widths):
        bw_data = valid_df[valid_df['bit_width_category'] == bit_width]
        axes[1].scatter(bw_data['total_params'], bw_data['final_val_loss'], 
                       color=colors[i], label=bit_width, s=100, alpha=0.7)
        
        # Add trend line if more than 1 point
        if len(bw_data) > 1:
            z = np.polyfit(bw_data['total_params'], bw_data['final_val_loss'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(bw_data['total_params'].min(), bw_data['total_params'].max(), 100)
            axes[1].plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Model Size (Parameters)')
    axes[1].set_ylabel('Final Validation Loss')
    axes[1].set_title('Validation Loss vs Model Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add parameter count annotations
    for i, row in valid_df.iterrows():
        axes[0].annotate(f'{row["total_params"]/1e6:.1f}M', 
                        (row['total_params'], row['final_train_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1].annotate(f'{row["total_params"]/1e6:.1f}M', 
                        (row['total_params'], row['final_val_loss']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('loss_vs_model_size_by_bitwidth.png', dpi=150, bbox_inches='tight')
    print("📊 Saved loss vs model size (by bit-width) to: loss_vs_model_size_by_bitwidth.png")
    plt.show()

def plot_loss_vs_model_size_by_hyperparams(df):
    """Graph 3: Same as #2 but with hyperparameter separation"""
    
    # Filter for runs with valid data
    valid_df = df.dropna(subset=['total_params', 'final_train_loss', 'final_val_loss'])
    
    if valid_df.empty:
        print("No valid data for loss vs model size with hyperparameters analysis")
        return
    
    # Create a combined grouping by bit-width and hyperparameters
    valid_df['combined_category'] = valid_df['bit_width_category'] + '_' + valid_df['hyperparams_signature']
    categories = valid_df['combined_category'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training and Validation Loss vs Model Size (Grouped by Bit-Width and Hyperparameters)', fontsize=16)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    
    # Main plots: Training and Validation Loss vs Model Size
    for i, category in enumerate(categories):
        cat_data = valid_df[valid_df['combined_category'] == category]
        
        # Training loss
        axes[0, 0].scatter(cat_data['total_params'], cat_data['final_train_loss'], 
                          color=colors[i], label=category[:30] + '...', s=100, alpha=0.7)
        
        # Validation loss
        axes[0, 1].scatter(cat_data['total_params'], cat_data['final_val_loss'], 
                          color=colors[i], label=category[:30] + '...', s=100, alpha=0.7)
    
    axes[0, 0].set_xlabel('Model Size (Parameters)')
    axes[0, 0].set_ylabel('Final Training Loss')
    axes[0, 0].set_title('Training Loss vs Model Size')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Model Size (Parameters)')
    axes[0, 1].set_ylabel('Final Validation Loss')
    axes[0, 1].set_title('Validation Loss vs Model Size')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hyperparameter analysis plots
    # Learning rate vs performance
    axes[1, 0].scatter(valid_df['learning_rate'], valid_df['final_train_loss'], 
                      c=valid_df['uniform_bit_width'], cmap='viridis', s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Final Training Loss')
    axes[1, 0].set_title('Training Loss vs Learning Rate (colored by bit-width)')
    cbar1 = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar1.set_label('Bit Width')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Batch size vs performance
    axes[1, 1].scatter(valid_df['batch_size'], valid_df['final_val_loss'], 
                      c=valid_df['uniform_bit_width'], cmap='viridis', s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Final Validation Loss')
    axes[1, 1].set_title('Validation Loss vs Batch Size (colored by bit-width)')
    cbar2 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar2.set_label('Bit Width')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_vs_model_size_by_hyperparams.png', dpi=150, bbox_inches='tight')
    print("📊 Saved loss vs model size (by hyperparams) to: loss_vs_model_size_by_hyperparams.png")
    plt.show()

def create_data_summary_table(df):
    """Create a summary table of the analyzed data"""
    print("\\n" + "="*80)
    print("DETAILED DATA SUMMARY")
    print("="*80)
    
    print(f"\\nTotal Runs Analyzed: {len(df)}")
    print(f"Runs with Training Loss: {df['final_train_loss'].notna().sum()}")
    print(f"Runs with Validation Loss: {df['final_val_loss'].notna().sum()}")
    print(f"Runs with Quantization: {df['quant_enabled'].sum()}")
    
    print("\\n--- Model Architecture Variations ---")
    print("Model Size Categories:")
    for size_cat in df['model_size_category'].unique():
        count = len(df[df['model_size_category'] == size_cat])
        params = df[df['model_size_category'] == size_cat]['total_params'].iloc[0]
        print(f"  {size_cat}: {count} runs ({params/1e6:.1f}M parameters)")
    
    print("\\n--- Quantization Variations ---")
    print("Bit Width Categories:")
    for bw_cat in df['bit_width_category'].unique():
        count = len(df[df['bit_width_category'] == bw_cat])
        if 'bit' in bw_cat:
            avg_bw = df[df['bit_width_category'] == bw_cat]['uniform_bit_width'].iloc[0]
            print(f"  {bw_cat}: {count} runs (avg {avg_bw:.1f} bits)")
        else:
            print(f"  {bw_cat}: {count} runs")
    
    print("\\n--- Hyperparameter Variations ---")
    print(f"Learning Rates: {sorted(df['learning_rate'].unique())}")
    print(f"Batch Sizes: {sorted(df['batch_size'].unique())}")
    print(f"Weight Decay: {sorted(df['weight_decay'].unique())}")
    print(f"Dropout: {sorted(df['dropout'].unique())}")
    
    print("\\n--- Performance Summary ---")
    if df['final_train_loss'].notna().any():
        print(f"Training Loss Range: {df['final_train_loss'].min():.4f} - {df['final_train_loss'].max():.4f}")
    if df['final_val_loss'].notna().any():
        print(f"Validation Loss Range: {df['final_val_loss'].min():.4f} - {df['final_val_loss'].max():.4f}")

def main():
    """Main analysis function"""
    print("🔍 Creating detailed model size and quantization analysis...")
    
    conn = sqlite3.connect('wandb_data.db')
    
    # Extract detailed configuration data
    print("\\n1. Extracting detailed configuration data...")
    df = extract_detailed_config_data(conn)
    
    # Create data summary
    create_data_summary_table(df)
    
    # Create the three requested graphs
    print("\\n2. Creating Graph 1: Loss vs Bit-Width (grouped by Model Size)...")
    plot_loss_vs_bitwidth_by_model_size(df)
    
    print("\\n3. Creating Graph 2: Loss vs Model Size (grouped by Bit-Width)...")
    plot_loss_vs_model_size_by_bitwidth(df)
    
    print("\\n4. Creating Graph 3: Loss vs Model Size (with Hyperparameter Separation)...")
    plot_loss_vs_model_size_by_hyperparams(df)
    
    # Save detailed data for further analysis
    df.to_csv('detailed_analysis_data.csv', index=False)
    print("\\n💾 Saved detailed analysis data to: detailed_analysis_data.csv")
    
    conn.close()
    print("\\n✅ Detailed analysis complete!")

if __name__ == "__main__":
    main()
