import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

def draw_scientific_architecture():
    # 1. Canvas Setup
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.axis('off')
    
    # 2. Font Configuration
    chinese_font = None
    possible_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Malgun Gothic', 'DengXian']
    import matplotlib.font_manager as fm
    for f in possible_fonts:
        found_fonts = [fp for fp in fm.findSystemFonts() if f.lower() in fp.lower()]
        if found_fonts:
            chinese_font = FontProperties(fname=found_fonts[0])
            break
    if chinese_font is None:
        chinese_font = FontProperties()

    # 3. Constants & Configuration
    CX = 50           # Central Axis X
    W_BLOCK = 30      # Standard Block Width
    H_BLOCK = 4.5     # Standard Block Height (Reduced)
    GAP_Y = 4         # Vertical Gap between blocks (Reduced)
    MARGIN = 2        # Margin for containers (Reduced)
    
    # Colors
    C_BACKBONE = '#E6F3FF'
    C_ATTN = '#FFF2CC'
    C_NECK = '#E2F0D9'
    C_HEAD = '#FCE4D6'
    C_EDGE = '#555555'

    # 4. Helper Functions
    def draw_box(x_center, y_center, w, h, text, color, subtext=None):
        # Calculate bottom-left corner from center
        x = x_center - w/2
        y = y_center - h/2
        
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=1.2, edgecolor=C_EDGE, facecolor=color)
        ax.add_patch(rect)
        
        ax.text(x_center, y_center + (1.2 if subtext else 0), text, ha='center', va='center', 
                fontproperties=chinese_font, fontsize=10, fontweight='bold', color='#333333')
        if subtext:
             ax.text(x_center, y_center - 1.5, subtext, ha='center', va='center', 
                fontproperties=chinese_font, fontsize=8, color='#555555')
        
        # Return cardinal points for connections: (top, bottom, left, right)
        return {
            'top': (x_center, y + h),
            'bottom': (x_center, y),
            'left': (x, y_center),
            'right': (x + w, y_center),
            'center': (x_center, y_center)
        }

    def draw_arrow(p1, p2, style='->', connection='arc3,rad=0'):
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle=style, color=C_EDGE, lw=1.2, connectionstyle=connection))

    def draw_container(y_top, y_bottom, label):
        x_left = 5
        x_right = 95
        h = y_top - y_bottom
        rect = patches.Rectangle((x_left, y_bottom), x_right - x_left, h, 
                                 linewidth=1, edgecolor='#AAAAAA', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x_left + 1, y_top - 1.5, label, fontproperties=chinese_font, 
                fontsize=9, color='gray', ha='left', va='top')

    # 5. Layout Execution
    # Define Y-coordinates for centers of each rank
    # Start from top
    Y_START = 100
    
    # Ranks
    y_input = Y_START
    y_stem = y_input - (H_BLOCK + GAP_Y)
    y_p2 = y_stem - (H_BLOCK + GAP_Y)
    y_p3 = y_p2 - (H_BLOCK + GAP_Y)
    y_p4 = y_p3 - (H_BLOCK + GAP_Y)
    
    y_sppf = y_p4 - (H_BLOCK + GAP_Y * 1.5) # Extra gap for section break
    y_fusion = y_sppf - (H_BLOCK + GAP_Y + H_BLOCK/2) # Fusion is taller
    
    y_head = y_fusion - (H_BLOCK + GAP_Y * 1.5 + H_BLOCK/2)
    
    # --- Input ---
    box_input = draw_box(CX, y_input, W_BLOCK, H_BLOCK, "输入图像", "white", "(640x640x3)")
    
    # --- Backbone ---
    # Container boundary calculation
    bb_top = y_stem + H_BLOCK/2 + MARGIN
    bb_bottom = y_p4 - H_BLOCK/2 - MARGIN
    
    box_stem = draw_box(CX, y_stem, W_BLOCK, H_BLOCK, "Stem 模块", C_BACKBONE)
    box_p2 = draw_box(CX, y_p2, W_BLOCK, H_BLOCK, "P2 特征层", C_BACKBONE, "FasterNet Block")
    box_p3 = draw_box(CX, y_p3, W_BLOCK, H_BLOCK, "P3 特征层", C_BACKBONE, "FasterNet Block")
    box_p4 = draw_box(CX, y_p4, W_BLOCK, H_BLOCK, "P4 特征层", C_BACKBONE, "FasterNet Block")
    
    # Attention Modules (Right side)
    x_attn = CX + W_BLOCK/2 + 10 # Reduced offset
    box_p2_attn = draw_box(x_attn, y_p2, 14, H_BLOCK, "CBAM", C_ATTN, "ECA")
    box_p3_attn = draw_box(x_attn, y_p3, 14, H_BLOCK, "CBAM", C_ATTN, "CA")
    box_p4_attn = draw_box(x_attn, y_p4, 14, H_BLOCK, "CBAM", C_ATTN, "ECA")
    
    draw_container(bb_top, bb_bottom, "骨干网络 (RiceFasterNet)")
    
    # Connections Backbone
    draw_arrow(box_input['bottom'], box_stem['top'])
    draw_arrow(box_stem['bottom'], box_p2['top'])
    draw_arrow(box_p2['bottom'], box_p3['top'])
    draw_arrow(box_p3['bottom'], box_p4['top'])
    
    # Attn connections
    draw_arrow(box_p2['right'], box_p2_attn['left'])
    draw_arrow(box_p3['right'], box_p3_attn['left'])
    draw_arrow(box_p4['right'], box_p4_attn['left'])
    
    # --- Neck ---
    neck_top = y_sppf + H_BLOCK/2 + MARGIN
    neck_bottom = y_fusion - H_BLOCK*0.9 - MARGIN # Fusion is approx 1.8x height
    
    box_sppf = draw_box(CX, y_sppf, W_BLOCK, H_BLOCK, "SPPF 池化", C_NECK)
    
    # Fusion block is taller
    box_fusion = draw_box(CX, y_fusion, W_BLOCK, H_BLOCK*1.8, "特征融合\n(FPN + PAN)", C_NECK, "自顶向下 + 自底向上")
    
    draw_container(neck_top, neck_bottom, "颈部网络 (RiceYOLONeck)")
    
    # Connections Neck
    draw_arrow(box_p4['bottom'], box_sppf['top'])
    draw_arrow(box_sppf['bottom'], box_fusion['top'])
    
    # Cross connections (Attn -> Fusion)
    # P3 Attn -> Fusion Top Half
    p3_out = box_p3_attn['right']
    # Target: Right side of Fusion, upper part
    fusion_in_p3 = (box_fusion['right'][0], box_fusion['top'][1] - H_BLOCK*0.4)
    
    ax.annotate('', xy=fusion_in_p3, xytext=p3_out,
                arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.2, connectionstyle="arc3,rad=0.5"))

    # P2 Attn -> Fusion Bottom Half
    p2_out = box_p2_attn['right']
    # Target: Right side of Fusion, lower part
    fusion_in_p2 = (box_fusion['right'][0], box_fusion['bottom'][1] + H_BLOCK*0.4)
    
    # This needs a longer curve
    ax.annotate('', xy=fusion_in_p2, xytext=p2_out,
                arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.2, connectionstyle="arc3,rad=0.6"))

    # --- Head ---
    head_top = y_head + H_BLOCK/2 + MARGIN
    head_bottom = y_head - H_BLOCK/2 - MARGIN
    
    w_head = 22
    offset_head = 13 # Reduced offset
    box_cls = draw_box(CX - offset_head, y_head, w_head, H_BLOCK, "分类分支", C_HEAD, "nc-class logits")
    box_reg = draw_box(CX + offset_head, y_head, w_head, H_BLOCK, "回归分支", C_HEAD, "4 * reg_max DFL")
    
    draw_container(head_top, head_bottom, "解耦检测头 (Decoupled Head)")
    
    # Connections Head
    # From Fusion Bottom Center to Heads
    # Use diagonal lines
    draw_arrow(box_fusion['bottom'], box_cls['top'])
    draw_arrow(box_fusion['bottom'], box_reg['top'])
    
    # 6. Save
    output_path = 'results/architecture_diagram_mpl.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    draw_scientific_architecture()
