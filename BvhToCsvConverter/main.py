import os

from bvhtoolbox import Bvh


def convert_bvh_to_csv(bvh_file, csv_file):
    with open(bvh_file, 'r') as f:
        bvh_data = Bvh(f.read())

    # Extract joint names
    joint_names = bvh_data.get_joints_names()

    # Get motion data
    motion_data = bvh_data.frames

    # Open CSV file for writing
    with open(filename.replace(".bvh", ".csv"), "w") as csv:
        # Write header with joint names
        csv.write(','.join(joint_names) + '\n')

        # Iterate through motion data frames
        for frame_data in motion_data:
            # Write joint angles to CSV
            csv.write(','.join(map(str, frame_data)) + '\n')


bvh_dir = 'E:/Datasety/kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH'

for folder in os.listdir(bvh_dir):
    for filename in os.listdir(os.path.join(bvh_dir, folder)):
        if filename.endswith('.bvh'):
            bvh_file_path = os.path.join(bvh_dir, folder, filename)
            csv_output_path = os.path.join('D:/ML/Diploma_work/data', filename.replace('.bvh', '.csv'))
            convert_bvh_to_csv(bvh_file_path, csv_output_path)
            print(f"Converted {bvh_file_path} to {csv_output_path}")
