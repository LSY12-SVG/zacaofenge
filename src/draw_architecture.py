from graphviz import Digraph
import os

def draw_training_pipeline():
    dot = Digraph(comment='Training Pipeline Overview', format='png')
    
    # Global Attributes: Wide, Scientific, Clean
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.8', compound='true')
    dot.attr('node', shape='rect', style='rounded,filled', fontname='Arial', fontsize='12', margin='0.15,0.1')
    dot.attr('edge', fontname='Arial', fontsize='10', penwidth='1.2', arrowsize='0.7')

    # Color Palette
    c_data = '#E6F3FF'    # Light Blue
    c_train = '#FFF2CC'   # Light Yellow
    c_opt = '#E2F0D9'     # Light Green
    c_log = '#FCE4D6'     # Light Orange
    c_store = '#E1E1E1'   # Light Gray

    # --- Cluster 1: Data Preparation ---
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Preparation', style='dashed', color='gray')
        
        c.node('Dataset', 'Dataset\n(YOLO Format)', fillcolor=c_data, shape='folder')
        
        # Albumentations list as an HTML-like label for detail
        aug_label = '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Albumentations</B></TD></TR>
            <TR><TD ALIGN="LEFT">- Rotate / Flip</TD></TR>
            <TR><TD ALIGN="LEFT">- Affine / RandomCrop</TD></TR>
            <TR><TD ALIGN="LEFT">- Brightness / Contrast</TD></TR>
            <TR><TD ALIGN="LEFT">- Normalize</TD></TR>
            </TABLE>
        >'''
        c.node('Augment', aug_label, fillcolor=c_data, shape='component')
        
        c.edge('Dataset', 'Augment', label='Load')

    # --- Cluster 2: Training Core ---
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='Training Loop', style='dashed', color='gray')
        
        c.node('Model', 'RiceYOLOv8s\n(Custom Model)', fillcolor=c_train, shape='box3d')
        
        # Optimization Group
        with c.subgraph(name='cluster_opt') as c_opt_group:
            c_opt_group.attr(label='', style='invis')
            c_opt_group.node('Optimizer', 'AdamW\nOptimizer', fillcolor=c_opt)
            c_opt_group.node('Scheduler', 'CosineAnnealingLR\nScheduler', fillcolor=c_opt)
            c_opt_group.edge('Scheduler', 'Optimizer', label='Adjust LR')
        
        # Train Step
        c.edge('Optimizer', 'Model', label='Update Weights')
        c.edge('Model', 'Optimizer', label='Gradients', dir='back')

    # --- Cluster 3: MLOps & Artifacts ---
    with dot.subgraph(name='cluster_ops') as c:
        c.attr(label='Monitoring & Artifacts', style='dashed', color='gray')
        
        c.node('MLflow', 'MLflow Tracking\n(Real-time)', fillcolor=c_log, shape='note')
        c.node('Checkpoints', 'Checkpoints\n(.pt files)', fillcolor=c_store, shape='cylinder')
        c.node('Curves', 'Training Curves\n(Loss/Metrics)', fillcolor=c_store, shape='note')
        
        c.edge('MLflow', 'Curves', style='dashed')

    # --- Main Flow Connections ---
    dot.edge('Augment', 'Model', label='Batch (B, C, H, W)')
    
    dot.edge('Model', 'MLflow', label='Log Metrics')
    dot.edge('Model', 'Checkpoints', label='Save Best')

    # Save
    output_path = os.path.join('results', 'training_pipeline')
    if not os.path.exists('results'):
        os.makedirs('results')
        
    try:
        dot.render(output_path, view=False)
        print(f"Diagram generated at: {output_path}.png")
    except Exception as e:
        print(f"Error rendering graphviz: {e}")
        print("Source saved.")

if __name__ == '__main__':
    draw_training_pipeline()
