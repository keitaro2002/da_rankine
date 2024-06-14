import os

from src.config import config
import numpy as np

def rename_files_in_directory(directory):
    # ディレクトリ内の全てのファイルをリストアップ
    files = os.listdir(directory)

    # ファイル名が 'rwindvalues_analysis' で始まるものをフィルタリング
    files_to_rename = [f for f in files if f.startswith('rwindvalues_analysis')]

    for filename in files_to_rename:
        # 新しいファイル名を作成
        new_filename = filename.replace('rwindvalues_analysis', 'windvalues_analysis', 1)
        
        # フルパスを作成
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        # ファイル名を変更
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} -> {new_file_path}')

# # 使用例
# directory_path = "/data10/kinuki/da_rankine/results/analysis"
# rename_files_in_directory(directory_path)


def combine_files(directory, start, end, interval):
    all_u_x = []
    all_u_y = []
    noise_flag = config.noise_flag

    for i in range(start, end, interval):
        p_range_start = i
        p_range_end = min(i + interval - 1, end - 1)

        noise_part = '' if noise_flag else '_no_noise'
        filename = f'windvalues_analysis{noise_part}_{p_range_start}_{p_range_end}.npz'

        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f'Error: Missing file {filename}')
            return

        data = np.load(filepath)
        all_u_x.append(data['u_x'])
        all_u_y.append(data['u_y'])

    combined_u_x = np.concatenate(all_u_x, axis=1)
    combined_u_y = np.concatenate(all_u_y, axis=1)

    combined_data = {'u_x': combined_u_x, 'u_y': combined_u_y}
    
    np.savez(os.path.join(directory, f'combined_windvalues_analysis{noise_part}.npz'), **combined_data)
    print(f'Files combined successfully into combined_windvalues_analysis{noise_part}.npz')

# 使用例
directory_path = "/data10/kinuki/da_rankine/results/analysis"
combine_files(directory_path, 0, 91 * 91, 50)

