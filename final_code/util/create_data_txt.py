import os
import pandas as pd

# Adjust paths as needed
image_folder = '/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/d/align_data'
csv_file = '/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/deepl_final_6_18_v5.csv'
output_file = os.path.join(image_folder, 'dataset_train.txt')

# Load CSV file
try:
    data = pd.read_csv(csv_file)
    print(f"CSV file '{csv_file}' loaded successfully.")
except Exception as e:
    print(f"Failed to load CSV file '{csv_file}'. Error: {e}")
    exit(1)

try:
    with open(output_file, 'w') as f:
        for index, row in data.iterrows():
            img_name = row['#name']
            pose = row[['x', 'y', 'alt', 'heading', 'pitch', 'roll']].tolist()
            pose_str = ' '.join(map(str, pose))
            f.write(f"{img_name} {pose_str}\n")
            if index % 100 == 0:
                print(f"Processed {index} rows")
    print("Text file created successfully.")
except Exception as e:
    print(f"Failed to create text file '{output_file}'. Error: {e}")
