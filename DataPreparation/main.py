import os

import matplotlib.pyplot as plt
import pandas as pd


#
def transform_to_mediapipe_format(row, mapping):
    return [[idx, row[f"{column}.X"], row[f"{column}.Y"]] for column, idx in mapping.items()]


def transform_and_save_dataset(file_path, output_path, left_mapping, right_mapping):
    data = pd.read_csv(file_path)
    # Determine which mapping to use based on the file name
    mapping = left_mapping if 'left' in file_path.lower() else right_mapping
    transformed_data = data.apply(lambda row: transform_to_mediapipe_format(row, mapping), axis=1)
    flattened_data = pd.DataFrame([item for sublist in transformed_data for item in sublist],
                                  columns=['LandmarkIndex', 'X', 'Y'])
    flattened_data.to_csv(output_path, index=False)


bvh_dir = "E:/Datasety/kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH"
csv_dir = "D:/ML/Diploma_work/DataPreparation/data"
os.makedirs(csv_dir, exist_ok=True)

left_hand_mapping = {
    'LeftHand': 0,  # Wrist
    'LeftHandThumb1': 1,
    'LeftHandThumb2': 2,
    'LeftHandThumb3': 3,
    'LeftHandThumb3End': 4,  # Thumb tip
    'LeftHandIndex1': 5,
    'LeftHandIndex2': 6,
    'LeftHandIndex3': 7,
    'LeftHandIndex3End': 8,  # Index finger tip
    'LeftHandMiddle1': 9,
    'LeftHandMiddle2': 10,
    'LeftHandMiddle3': 11,
    'LeftHandMiddle3End': 12,  # Middle finger tip
    'LeftHandRing1': 13,
    'LeftHandRing2': 14,
    'LeftHandRing3': 15,
    'LeftHandRing3End': 16,  # Ring finger tip
    'LeftHandPinky1': 17,
    'LeftHandPinky2': 18,
    'LeftHandPinky3': 19,
    'LeftHandPinky3End': 20
}

right_hand_mapping = {
    'RightHand': 0,  # Wrist
    'RightHandThumb1': 1,
    'RightHandThumb2': 2,
    'RightHandThumb3': 3,
    'RightHandThumb3End': 4,  # Thumb tip
    'RightHandIndex1': 5,
    'RightHandIndex2': 6,
    'RightHandIndex3': 7,
    'RightHandIndex3End': 8,  # Index finger tip
    'RightHandMiddle1': 9,
    'RightHandMiddle2': 10,
    'RightHandMiddle3': 11,
    'RightHandMiddle3End': 12,  # Middle finger tip
    'RightHandRing1': 13,
    'RightHandRing2': 14,
    'RightHandRing3': 15,
    'RightHandRing3End': 16,  # Ring finger tip
    'RightHandPinky1': 17,
    'RightHandPinky2': 18,
    'RightHandPinky3': 19,
    'RightHandPinky3End': 20
}

for folder in os.listdir(bvh_dir):
    folder_path = os.path.join(bvh_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            for hand in ['Left', 'Right']:
                hand_df = df.iloc[:, 25:99] if hand == 'Left' else df.iloc[:, 109:183]
                hand_df = hand_df[hand_df.columns.drop(list(hand_df.filter(regex='.Z')))]
                hand_df = hand_df.drop(hand_df.filter(regex='InHand').columns, axis=1)
                hand_filename = filename.replace(".csv", f"_{hand.lower()}_hand.csv")
                hand_df.to_csv(os.path.join(csv_dir, hand_filename), index=False)

                transformed_file_path = os.path.join(csv_dir, hand_filename.replace(".csv", "_mediapipe.csv"))

                # Call the function with both mappings
                transform_and_save_dataset(os.path.join(csv_dir, hand_filename),
                                           transformed_file_path,
                                           left_hand_mapping,
                                           right_hand_mapping)

            print(f"Processed {filename}")

emotion_mapping = {
    'A': 'Angry',
    'D': 'Disgust',
    'F': 'Fearful',
    'H': 'Happy',
    'N': 'Neutral',
    'SA': 'Sad',
    'SU': 'Surprise',
}
for filename in os.listdir(csv_dir):
    if filename.endswith('_mediapipe.csv'):
        file_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(file_path)

        if len(filename) >= 6 and filename[3:5].isalpha():
            emotion_code = filename[3:5]
        elif len(filename) >= 5 and filename[3].isalpha():
            emotion_code = filename[3]
        else:
            emotion_code = 'Unknown'
        if emotion_code == 'A':
            emotion = emotion_mapping.get('A', 'Unknown')
        elif emotion_code == 'D':
            emotion = emotion_mapping.get('D', 'Unknown')
        elif emotion_code == 'F':
            emotion = emotion_mapping.get('F', 'Unknown')
        elif emotion_code == 'H':
            emotion = emotion_mapping.get('H', 'Unknown')
        elif emotion_code == 'N':
            emotion = emotion_mapping.get('N', 'Unknown')
        elif emotion_code == 'SA':
            emotion = emotion_mapping.get('SA', 'Unknown')
        elif emotion_code == 'SU':
            emotion = emotion_mapping.get('SU', 'Unknown')
        else:
            emotion = 'Unknown'

        emotion = emotion_mapping.get(emotion_code, 'Unknown')

        df['Emotion'] = emotion

        df.to_csv(file_path, index=False)
        print(f"Added Emotion: {emotion} to {filename}")
print("done")
print("Processing complete")

file_list = [file for file in os.listdir(csv_dir) if file.endswith('_mediapipe.csv')]

merged_data = pd.DataFrame()

for file in file_list:
    file_path = os.path.join(csv_dir, file)
    data = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, data])
    print(f"Processed {file}")

merged_data.to_csv('merged_data.csv', index=False)
print("done")

data = pd.read_csv('D:/ML/Diploma_work/DataPreparation/merged_data.csv')

sampled_data = data.sample(frac=0.05, random_state=42)

stratified_sample = data.groupby('Emotion', group_keys=False).apply(lambda x: x.sample(frac=0.3))

sampled_data.to_csv('sampled_data.csv', index=False)

# Load the data
data = pd.read_csv('D:\ML\Diploma_work\DataPreparation\merged_data.csv')

# Create a directory to save the heatmaps
save_dir = 'E:/Datasety/heatmaps'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Number of joints per hand
num_joints = 21

# Iterate through each hand position and plot a heatmap
for i in range(0, len(data), num_joints):
    # Reshape the X and Y coordinates for each hand position
    x_coords = data.iloc[i:i + num_joints]['X'].values
    y_coords = data.iloc[i:i + num_joints]['Y'].values
    emotion = data.iloc[i]['Emotion']

    # Generate a heatmap
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c=range(num_joints), cmap='hot', s=100)
    plt.colorbar(label='Landmark Index')
    plt.title(f'Heatmap of Hand Joints Position for {emotion} - Sample {i // num_joints}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()

    # Save the figure
    plt.savefig(os.path.join(save_dir, f'heatmap_{emotion}_{i // num_joints}.png'))
    print(f"Processed heatmap for {emotion} - Sample {i // num_joints}")
    plt.close()  # Close the plot to free memory
