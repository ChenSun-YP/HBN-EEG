import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_cnn_architecture_matplotlib():
    # --- Setup ---
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_aspect('equal')
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 30)
    ax.axis('off') # Hide axes

    # --- Configuration ---
    block_width = 7
    block_height = 4
    # The spacing should be used *after* calling draw_block to move to the next position
    default_spacing = 1.5 
    start_y = 28
    
    # Define colors
    COLOR_INPUT = '#f9f'
    COLOR_CONV = '#add8e6'
    COLOR_POOL_PROJ = '#ff9'
    COLOR_OUTPUT = '#cfc'
    # COLOR_BLOCK = 'lightgray' # This was commented out in previous version too.

    # --- Drawing Helper Function ---
    def draw_block(x, y, width, height, label, tensor_shape, color, fill_color, alpha=0.9):
        # Draw the main block rectangle
        rect = patches.Rectangle((x, y - height), width, height, 
                                 edgecolor=color, facecolor=fill_color, alpha=alpha, linewidth=2, zorder=2)
        ax.add_patch(rect)
        
        # Add the layer label
        ax.text(x + width / 2, y - height / 2, label, 
                ha='center', va='center', fontsize=9, fontweight='bold', wrap=True)

        # Add the tensor shape label above the block
        if tensor_shape:
            ax.text(x + width / 2, y + 0.3, tensor_shape, 
                    ha='center', va='bottom', fontsize=8, color='dimgray')
        
        # This function returns the y-coordinate of the bottom of the drawn block
        return y - height

    # --- Draw Architecture ---

    y_pos = start_y # Current top-y position for the next block

    # 1. Input
    y_block_bottom = draw_block(
        x=0.5, y=y_pos, width=block_width, height=1.5,
        label='Input Tensor', tensor_shape='(B, C, T)', 
        color='black', fill_color=COLOR_INPUT
    )
    y_pos = y_block_bottom - default_spacing # Update y_pos for next block's top
    
    # Add connecting arrow
    ax.annotate('', xy=(4, y_block_bottom + default_spacing / 2), xytext=(4, y_block_bottom - 0.5), # Adjust arrow positions
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # 2. Temporal Convolution Block (Grouped)
    temporal_block_label = (
        'Temporal Convolution Block\n'
        '1. Conv1d(C -> 64, k=25) -> BN -> ReLU -> Dropout\n'
        '2. Conv1d(64 -> 128, k=13) -> BN -> ReLU -> Dropout\n'
        '3. Conv1d(128 -> 256, k=7) -> BN -> ReLU'
    )
    y_block_bottom = draw_block(
        x=0, y=y_pos, width=block_width + 1, height=block_height * 2.5,
        label=temporal_block_label, tensor_shape='(B, 64/128/256, T)', 
        color='black', fill_color=COLOR_CONV
    )
    y_pos = y_block_bottom - default_spacing

    # Add connecting arrow
    ax.annotate('', xy=(4, y_block_bottom + default_spacing / 2), xytext=(4, y_block_bottom - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

    # 3. Spatial Convolution Block (Grouped)
    spatial_block_label = (
        'Dense Feature Mixing Block (Spatial Convolutions)\n'
        '1. Conv1d(256 -> 256, k=3) -> ReLU -> Dropout\n'
        '2. Conv1d(256 -> 256, k=3) -> ReLU'
    )
    y_block_bottom = draw_block(
        x=0.5, y=y_pos, width=block_width, height=block_height * 1.5,
        label=spatial_block_label, tensor_shape='(B, 256, T)', 
        color='black', fill_color=COLOR_CONV
    )
    y_pos = y_block_bottom - default_spacing

    # Add connecting arrow
    ax.annotate('', xy=(4, y_block_bottom + default_spacing / 2), xytext=(4, y_block_bottom - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

    # 4. Global Pooling & Projection
    # This block has sub-components, so we'll manage y_pos more granularly

    # Pooling Layer
    # y_pos is currently the top of where the pooling block should start
    pool_block_height = block_height / 3
    y_block_bottom = draw_block(
        x=1, y=y_pos, width=block_width - 2, height=pool_block_height,
        label='Global Pooling (AdaptiveAvgPool1d(1))', tensor_shape='(B, 256, T)',
        color='black', fill_color=COLOR_POOL_PROJ
    )
    
    # Text for squeeze operation, placed between pooling and linear
    squeeze_text_y = y_block_bottom - (default_spacing / 2) # Place it roughly in the middle of the gap
    ax.text(4, squeeze_text_y, '(B, 256, 1) -> Squeeze -> (B, 256)',
            ha='center', va='center', fontsize=8, color='dimgray')
    
    y_pos = y_block_bottom - default_spacing # Update y_pos for linear layer
    
    # Linear Layer
    linear_block_height = block_height / 3
    y_block_bottom = draw_block(
        x=1, y=y_pos, width=block_width - 2, height=linear_block_height,
        label='Final Projection (Linear: 256 -> H)', tensor_shape='(B, 256)',
        color='black', fill_color=COLOR_POOL_PROJ
    )
    y_pos = y_block_bottom - default_spacing

    # Add connecting arrow from linear to output
    ax.annotate('', xy=(4, y_block_bottom + default_spacing / 2), xytext=(4, y_block_bottom - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

    # 5. Output
    y_block_bottom = draw_block(
        x=0.5, y=y_pos, width=block_width, height=1.5,
        label='Output Embedding', tensor_shape='(B, H)',
        color='black', fill_color=COLOR_OUTPUT
    )
    
    # Final plot save
    plt.title('CNN Encoder Architecture (Matplotlib Render)', fontsize=14)
    
    # Save the figure
    plt.savefig('cnn_architecture_matplotlib.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Diagram 'cnn_architecture_matplotlib.png' generated successfully!")

# Call the function to create the diagram
create_cnn_architecture_matplotlib()