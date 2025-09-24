import numpy as np
import matplotlib.pyplot as plt
import os

# 定义文件路径和温度
temperatures = [1270,1770,2270,2770,3270]
file_dir = '/home/remote/vasp/nep-d3/new/press/results'

# 初始化存储数据的字典
cu_data = {}
mo_data = {}

# 读取所有文件的数据
for temp in temperatures:
    file_path = os.path.join(file_dir, f'press_{temp}.txt')
    data = np.loadtxt(file_path, skiprows=1)
    time = data[:, 0]
    cu_msd = data[:, 1]
    mo_msd = data[:, 2]
    
    cu_data[temp] = (time, cu_msd)
    mo_data[temp] = (time, mo_msd)

# 选择标注点的索引（例如每50个点标注一个）
#annotation_indices = [0, 5, 10, 15, 20]

# 绘制铜的扩散趋势图
plt.figure(figsize=(6.3, 3.54))
for temp in temperatures:
    time, cu_msd = cu_data[temp]
    line, = plt.plot(time, cu_msd, label=f'{temp} bar')
    
    # 标注选定的点
   # for i in annotation_indices:
    #    if i < len(time):
     #       plt.annotate(f'{cu_msd[i]:.2f}', (time[i], cu_msd[i]), 
      #                  textcoords="offset points", xytext=(0,10), ha='center',
       #                 fontsize=8, color=line.get_color())

plt.xlabel('Time (ps)')
plt.ylabel('MSD (A^2)')
plt.title('Cu Diffusion Trends at Different Pressures')
plt.legend()
plt.grid(True)
plt.savefig('/home/remote/vasp/nep-d3/new/press/results/cu_diffusion_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制钼的扩散趋势图
plt.figure(figsize=(6.3, 3.54))
for temp in temperatures:
    time, mo_msd = mo_data[temp]
    line, = plt.plot(time, mo_msd, label=f'{temp} bar')
    
    # 标注选定的点
 #   for i in annotation_indices:
  #      if i < len(time):
   #         plt.annotate(f'{mo_msd[i]:.2f}', (time[i], mo_msd[i]), 
    #                    textcoords="offset points", xytext=(0,10), ha='center',
     #                   fontsize=8, color=line.get_color())

plt.xlabel('Time (ps)')
plt.ylabel('MSD (A^2)')
plt.title('Mo Diffusion Trends at Different Pressures')
plt.legend()
plt.grid(True)
plt.savefig('/home/remote/vasp/nep-d3/new/press/results/mo_diffusion_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print('扩散趋势图已保存到 /home/remote/vasp/nep-d3/new/press/results 目录下')
