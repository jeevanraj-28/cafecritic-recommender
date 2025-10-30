# src/visualization/plots.py
import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    """
    Save plot to results/figures/ AND return fig for display
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    save_path = os.path.join(project_root, 'results', 'figures', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    
    return fig  # Return so notebook can display it