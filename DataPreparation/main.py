import glob
import os

import pandas as pd

bvh_dir = "BVH"
csv_dir = "DataPreparation/data"

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

for folder in os.listdir(bvh_dir):
    folder_path = os.path.join(bvh_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))

            df_left_hand = df.iloc[:, 25:99]
            df_left_hand = df_left_hand[df_left_hand.columns.drop(list(df_left_hand.filter(regex='.Z')))]

            df_right_hand = df.iloc[:, 109:183]
            df_right_hand = df_right_hand[df_right_hand.columns.drop(list(df_right_hand.filter(regex='.Z')))]

            right_hand_filename = filename.replace(".csv", "_right_hand.csv")
            left_hand_filename = filename.replace(".csv", "_left_hand.csv")

            df_right_hand.to_csv(os.path.join(csv_dir, right_hand_filename), index=False)
            df_left_hand.to_csv(os.path.join(csv_dir, left_hand_filename), index=False)

            print(f"Converted {filename} to {right_hand_filename} and {left_hand_filename}")

emotion_mapping = {
    'A': 'Angry',
    'D': 'Disgust',
    'F': 'Fearful',
    'H': 'Happy',
    'N': 'Neutral',
    'SA': 'Sad',
    'SU': 'Surprise',
}

all_data = []

for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
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
# print(features)
# print(labels)

left_hand_files_pattern = os.path.join(csv_dir, "*left_hand.csv")
left_hand_files = glob.glob(left_hand_files_pattern)

df_list = [pd.read_csv(file) for file in left_hand_files]
left_hand_ds = pd.concat(df_list, ignore_index=True)

left_hand_ds.rename(columns=lambda x: x.replace('LeftHand', ''), inplace=True)
left_hand_ds.rename(columns=lambda x: x.replace('LeftInHand', ''), inplace=True)
print("done")

right_hand_files_pattern = os.path.join(csv_dir, "*right_hand.csv")
right_hand_files = glob.glob(right_hand_files_pattern)

df_list = [pd.read_csv(file) for file in right_hand_files]
right_hand_ds = pd.concat(df_list, ignore_index=True)

right_hand_ds.rename(columns=lambda x: x.replace('RightHand', ''), inplace=True)
right_hand_ds.rename(columns=lambda x: x.replace('RightInHand', ''), inplace=True)
print("done")

ds = pd.concat([left_hand_ds, right_hand_ds], axis=0, ignore_index=True)
# filename = filename.replace(".csv", "_all_hands.csv")
# ds.to_csv(os.path.join(csv_dir, filename), index=False)

print("ds.shape", ds.shape)
