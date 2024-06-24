import numpy as np
from scipy.spatial.transform import Rotation as R

# # 변환 행렬
# transform_matrix = np.array([
#     [-0.10158499, -0.0204423,  0.0192721,  -7.92273845],
#     [-0.12908179, -0.08904014, 0.04370393,  1.5485332 ],
#     [ 0.24091402,  0.02519702, -0.11671822, 3.03941279],
#     [ 0.0,         0.0,         0.0,        1.0       ]
# ])

# # 스케일링 요소
# scale_factor = 1

# # A의 좌표
# coordinates_A = np.array([1.671764553, -1.233742623, 2.317066387, -41.66632456, 85.90357165, 50.53105859])

# def transform_coordinates(transform_matrix, coordinates, scale_factor):
#     # 스케일링 적용
#     scaled_transform_matrix = transform_matrix.copy()
#     scaled_transform_matrix[:3, :3] *= scale_factor

#     # 위치 좌표 [x, y, z] 변환
#     coord_homogeneous = np.append(coordinates[:3], 1)  # [x, y, z, 1]
#     transformed_coord = np.dot(scaled_transform_matrix, coord_homogeneous)
#     transformed_xyz = transformed_coord[:3]

#     # 회전 좌표 [heading, pitch, roll] 변환
#     r = R.from_euler('xyz', coordinates[3:], degrees=True)
#     rotation_matrix = scaled_transform_matrix[:3, :3]
#     transformed_rotation = r * R.from_matrix(rotation_matrix)
#     transformed_hpr = transformed_rotation.as_euler('xyz', degrees=True)

#     return np.concatenate((transformed_xyz, transformed_hpr))

# # 변환 적용
# transformed_coordinates = transform_coordinates(transform_matrix, coordinates_A, scale_factor)

# # B의 좌표 (예시 좌표)
# coordinates_B = np.array([3.780365690583588, -3.664146068, 1.289935110437421, 9.994622681504412, 84.69986287824241, 7.423498852966357])

# # 결과 비교
# print("Transformed Coordinates: ", transformed_coordinates)
# print("Expected Coordinates B: ", coordinates_B)
# print("Comparison with B dataset: ", np.allclose(transformed_coordinates, coordinates_B, atol=1e-2))

# 회전 행렬을 오일러 각도로 변환하는 함수
def eul2rot(heading, pitch, roll):
    # Heading (yaw), Pitch, Roll을 회전 행렬로 변환하는 함수
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

def create_pose_matrix(x, y, z, heading, pitch, roll):
    """Translation과 Euler 각도로부터 4x4 포즈 행렬을 생성합니다."""
    rot = eul2rot(heading, pitch, roll)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rot
    pose_matrix[:3, 3] = [x, y, z]
    return pose_matrix

# Example DataFrames with data (please ensure prame_data and sel_data are loaded properly)
# 데이터셋 각각에 대해 포즈 행렬 생성
team1_poses = [create_pose_matrix(row['x'], row['y'], row['z'], row['heading'], row['pitch'], row['roll']) for index, row in team1_data.iterrows()]
sim_poses = [create_pose_matrix(row['x'], row['y'], row['z'], row['heading'], row['pitch'], row['roll']) for index, row in sim_data.iterrows()]

# 각각의 변환 계산
transformations = []

for team1_pose, sim_pose in zip(team1_poses, sim_poses):
    # 각 쌍 (T_i * sim_pose = team1_pose)에 대해 변환 T_i 계산
    # T_i = team1_pose * inv(sim_pose) 형태로 재배열
    T_i = team1_pose @ inv(sim_pose)
    transformations.append(T_i)

# 변환 행렬들의 평균을 계산하여 최종 행렬 찾기
average_transformation = sum(transformations) / len(transformations)

print("Averaged Transformation Matrix T:\n", average_transformation)

# 회전 행렬에서 오일러 각도를 추출하는 함수
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([z, y, x])

import pandas as pd
import numpy as np


# team1 좌표를 sim 좌표계로 변환하는 함수
def transform_to_sim_coordinate(x_team1, y_team1, z_team1, heading_team1, pitch_team1, roll_team1, T):
    team1_position = np.vstack((x_team1, y_team1, z_team1, np.ones_like(x_team1)))
    sim_position = np.dot(T, team1_position)
    sim_positions = sim_position[:3, :].T

    sim_x = sim_positions[:, 0]
    sim_y = sim_positions[:, 1]
    sim_z = sim_positions[:, 2]

    R_team1 = np.array([eul2rot(h, p, r) for h, p, r in zip(heading_team1, pitch_team1, roll_team1)])
    R_sim = np.array([T[:3, :3] @ R for R in R_team1])
    sim_euler_angles = np.array([rotation_matrix_to_euler_angles(R) for R in R_sim])

    sim_heading = sim_euler_angles[:, 0]
    sim_pitch = sim_euler_angles[:, 1]
    sim_roll = sim_euler_angles[:, 2]

    return sim_x, sim_y, sim_z, np.rad2deg(sim_heading), np.rad2deg(sim_pitch), np.rad2deg(sim_roll)

# 변환 수행 및 정확도 계산 함수를 정의합니다.
def calculate_accuracy(team1_data, sim_poses_data, T):
    x_team1 = team1_data['x'].values
    y_team1 = team1_data['y'].values
    z_team1 = team1_data['z'].values
    heading_team1 = np.deg2rad(team1_data['heading'].values)
    pitch_team1 = np.deg2rad(team1_data['pitch'].values)
    roll_team1 = np.deg2rad(team1_data['roll'].values)
    
    sim_x, sim_y, sim_z, sim_heading, sim_pitch, sim_roll = transform_to_sim_coordinate(x_team1, y_team1, z_team1, heading_team1, pitch_team1, roll_team1, T)

    # 변환된 데이터를 DataFrame으로 만듭니다.
    sim_transformed_data = pd.DataFrame({
        'x': sim_x,
        'y': sim_y,
        'z': sim_z,
        'heading': sim_heading,
        'pitch': sim_pitch,
        'roll': sim_roll
    })

    # 변환된 데이터를 CSV 파일로 저장합니다.
    sim_transformed_data.to_csv('/content/drive/MyDrive/DeepL/final_csv/sim_transformed.csv', index=False)

    # 정답 데이터 불러오기
    sim_poses_x = sim_poses_data['x'].values
    sim_poses_y = sim_poses_data['y'].values
    sim_poses_z = sim_poses_data['z'].values
    sim_poses_heading = sim_poses_data['heading'].values
    sim_poses_pitch = sim_poses_data['pitch'].values
    sim_poses_roll = sim_poses_data['roll'].values

    # 각 좌표별 오차 계산
    error_x = np.abs(sim_x - sim_poses_x)
    error_y = np.abs(sim_y - sim_poses_y)
    error_z = np.abs(sim_z - sim_poses_z)
    error_heading = np.abs(sim_heading - sim_poses_heading)
    error_pitch = np.abs(sim_pitch - sim_poses_pitch)
    error_roll = np.abs(sim_roll - sim_poses_roll)
    
    # 오차의 평균값 계산
    avg_error = (error_x + error_y + error_z + error_heading + error_pitch + error_roll) / 6.0
    
    # 평균 오차를 기반으로 정확도 계산
    accuracy = 100.0 * (1.0 - avg_error.mean() / 100.0)
    
    return accuracy

# 정확도 계산
accuracy = calculate_accuracy(team1_data, sim_data, average_transformation)
print(f"Accuracy: {accuracy:.2f}")