import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def read_lammps_trajectory(filename):
    """Read LAMMPS trajectory file and extract atomic positions."""
    positions = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Parse the atomic positions
    # The atom coordinates start from line 9 (0-indexed as line 8)
    # Read exactly 6912 atoms as specified in the file header
    for i in range(9, 9 + 6912):
        if i >= len(lines):
            break
        line = lines[i]
        parts = line.strip().split()
        if len(parts) >= 5:  # Ensure we have enough data
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            positions.append([x, y, z])
    
    print(f"Number of atoms read: {len(positions)}")
    return np.array(positions)

def calculate_rdf(positions, box_size, dr=0.1, r_max=10.0):
    """Calculate radial distribution function."""
    # Calculate pairwise distances
    distances = pdist(positions, 'euclidean')
    
    # Create histogram
    r_bins = np.arange(0, r_max + dr, dr)
    hist, bin_edges = np.histogram(distances, bins=r_bins)
    
    # Calculate RDF
    r_vals = bin_edges[:-1] + dr/2
    volume = box_size[0] * box_size[1] * box_size[2]
    n_particles = len(positions)
    
    # Normalize
    for i in range(len(hist)):
        r_inner = r_vals[i] - dr/2
        r_outer = r_vals[i] + dr/2
        shell_volume = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        expected_density = n_particles / volume
        hist[i] = hist[i] / (expected_density * shell_volume)
    
    # 确保不出现负值
    hist = np.maximum(hist, 0)
    
    return r_vals, hist

def plot_rdf(r_vals, rdf):
    """Plot the radial distribution function."""
    plt.figure(figsize=(6.3, 3.54))
    # 创建插值函数以生成更平滑的曲线
    f = interp1d(r_vals, rdf, kind='cubic')
    
    # 生成更多点用于平滑曲线
    r_vals_smooth = np.linspace(r_vals.min(), r_vals.max(), 300)
    rdf_smooth = f(r_vals_smooth)
    
    # 确保插值后的值不小于0
    rdf_smooth = np.maximum(rdf_smooth, 0)
    
    # 将y轴值缩小10000倍
    rdf_smooth = rdf_smooth / 10000
    
    # 绘制平滑曲线
    plt.plot(r_vals_smooth, rdf_smooth, linewidth=2, linestyle='-', color=[189/255, 187/255, 189/255])
    
    # 找到2-4埃之间最大的两个峰并标注
    # 在原始数据上找峰，然后映射到平滑曲线上
    peaks, _ = find_peaks(rdf_smooth, height=0)
    if len(peaks) > 0:
        # 筛选出2-4埃之间的峰
        peak_positions = r_vals_smooth[peaks]
        mask = (peak_positions >= 2.0) & (peak_positions <= 4.0)
        valid_peaks = peaks[mask]
        
        if len(valid_peaks) > 0:
            # 找到2-4埃之间最大的两个峰
            peak_heights = rdf_smooth[valid_peaks]
            # 获取最大的两个峰的索引
            sorted_indices = np.argsort(peak_heights)[::-1]  # 降序排列
            top_two_indices = sorted_indices[:min(2, len(valid_peaks))]
            top_two_peaks = valid_peaks[top_two_indices]
            
            # 标注最大的两个峰
            for i, peak_idx in enumerate(top_two_peaks):
                peak_r = r_vals_smooth[peak_idx]
                peak_height = rdf_smooth[peak_idx]
                
                # 标注峰的位置
                plt.plot(peak_r, peak_height, 'ro', markersize=8)
                
                # 获取当前图表的坐标轴范围
                ax = plt.gca()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # 根据峰值位置调整注释文本位置
                if peak_r > xlim[1] - 1.0:
                    # 如果峰值靠近右边界，将文本放在左侧
                    text_x = peak_r - 0.8
                    plt.annotate(f'({peak_r:.2f}, {peak_height:.2f})', 
                                xy=(peak_r, peak_height), 
                                xytext=(text_x, peak_height + 0.05),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                                fontsize=9, ha='center')
                elif peak_height > ylim[1] - 0.1:
                    # 如果峰值靠近上边界，将文本放在下方
                    text_y = peak_height - 0.08
                    plt.annotate(f'({peak_r:.2f}, {peak_height:.2f})', 
                                xy=(peak_r, peak_height), 
                                xytext=(peak_r + 0.3, text_y),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                                fontsize=9, ha='center')
                else:
                    # 默认位置
                    plt.annotate(f'({peak_r:.2f}, {peak_height:.2f})', 
                                xy=(peak_r, peak_height), 
                                xytext=(peak_r + 0.3, peak_height + 0.05),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                                fontsize=9, ha='center')
    
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Mo Radial Distribution Function')
    plt.xticks(np.arange(0, int(r_vals_smooth.max()) + 1, 1))  # 设置x轴标注间隔为1
    # plt.grid(True)  # 移除网格线
    plt.savefig('rdf_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Read trajectory data
    positions = read_lammps_trajectory('final_config_stable.data')
    
    # Set box size from the file header
    box_size = [43.38, 43.38, 43.38]  # Box dimensions from file header
    
    # Calculate RDF
    r_vals, rdf = calculate_rdf(positions, box_size)
    
    # Plot RDF
    plot_rdf(r_vals, rdf)

if __name__ == "__main__":
    main()