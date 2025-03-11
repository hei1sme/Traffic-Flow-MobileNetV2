import pandas as pd
import numpy as np
from datetime import timedelta

def detect_downtime(df, time_col='Timestamp', threshold_short=300, threshold_long=1800):
    """
    Phát hiện downtime dựa trên khoảng cách giữa các timestamp.
    - Downtime ngắn (<5 phút) sẽ nội suy.
    - Downtime trung bình (5-30 phút) sẽ dùng mô hình dự đoán.
    - Downtime dài (>30 phút) sẽ đánh dấu 'No Data'.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    
    df['Time_Diff'] = df[time_col].diff().dt.total_seconds()
    df['Downtime_Type'] = 'Normal'
    
    df.loc[df['Time_Diff'] > threshold_short, 'Downtime_Type'] = 'Short'
    df.loc[df['Time_Diff'] > threshold_long, 'Downtime_Type'] = 'Long'
    
    return df

def handle_downtime(df, model=None, time_col='Timestamp', method='linear'):
    """
    Xử lý downtime theo loại:
    - Nội suy tuyến tính nếu downtime ngắn.
    - Dự đoán nếu downtime trung bình và có mô hình.
    - Đánh dấu 'No Data' nếu downtime dài.
    """
    df = detect_downtime(df, time_col)
    
    # Nội suy tuyến tính cho downtime ngắn
    df.interpolate(method=method, inplace=True)
    
    # Dự đoán downtime trung bình nếu có mô hình
    if model:
        missing_rows = df[df['Downtime_Type'] == 'Short']
        for index, row in missing_rows.iterrows():
            prev_data = df.iloc[index - 1].values[1:-2]  # Lấy dữ liệu trước downtime
            predicted_values = model.predict(prev_data.reshape(1, -1))
            df.iloc[index, 1:-2] = predicted_values  # Điền giá trị vào khoảng trống
    
    # Đánh dấu dữ liệu mất quá lâu
    df.loc[df['Downtime_Type'] == 'Long', df.columns[1:-2]] = np.nan
    df.fillna('No Data', inplace=True)
    
    return df

# Đọc dữ liệu
df = pd.read_csv("historical_data.csv")

# Xử lý downtime
df_cleaned = handle_downtime(df)

# Lưu lại dữ liệu đã xử lý
df_cleaned.to_csv("historical_data_cleaned.csv", index=False)