import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

def analyze_long_tail(file_path):
    # 1. 加载数据
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 读取数据，自动处理可能的空格
    df = pd.read_csv(file_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    # 2. 筛选数字列
    # 排除时间列（通常包含 'date' 或 'time' 字样）
    cols_to_exclude = ['date', 'rehh', 'hh']
    data_cols = [c for c in df.columns if not any(ex in c for ex in cols_to_exclude)]
    
    # 强制转换数据类型，无法转换的变 NaN 并删除
    numeric_df = df[data_cols].apply(pd.to_numeric, errors='coerce').dropna()

    if numeric_df.empty:
        print("错误：数据集中没有有效的数字列进行分析。")
        return

    # 3. 归一化 (Z-score)
    # 计算 (x - mean) / std，使不同量纲的数据在同一量级比较
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_df.columns)

    # 4. 计算偏度 (Skewness) 用于辅助判断长尾
    # Skewness > 0 表示右偏（长尾在右侧）
    skewness_values = numeric_df.skew().sort_values(ascending=False)

    # 5. 绘图准备：取绝对值并降序排列
    plt.figure(figsize=(14, 8), dpi=100)
    
    # 设置颜色映射，让图表不至于太乱
    colors = plt.get_cmap('tab20').resampled(len(data_cols))

    for i, col in enumerate(data_cols):
        # 核心逻辑：取绝对值 -> 排序 -> 翻转（从大到小）
        sorted_abs_data = np.sort(np.abs(scaled_df[col].values))[::-1]
        
        # 归一化 x 轴到 [0, 1] 区间，方便观察比例
        x_axis = np.linspace(0, 1, len(sorted_abs_data))
        
        plt.plot(x_axis, sorted_abs_data, label=f"{col} (Skew: {skewness_values[col]:.2f})", 
                 alpha=0.7, color=colors(i), linewidth=1.5)

    # 6. 图表装饰
    plt.title('Long-tail Distribution Analysis (Normalized & Sorted Absolute Values)', fontsize=14)
    plt.xlabel('Sample Percentage (0 to 1)', fontsize=12)
    plt.ylabel('Normalized Magnitude (|Z-score|)', fontsize=12)
    
    # 将图例放在右侧，按偏度大小排序显示会更有参考价值
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize='small', ncol=1)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存结果
    plt.savefig('tail_analysis_result.png', bbox_inches='tight')
    print("分析完成！图表已保存为 'tail_analysis_result.png'")
    plt.show()

# 执行分析
file_location = "dataset/DYG_new/DYG_new.csv"
analyze_long_tail(file_location)