import os

import pandas as pd
import numpy as np

bvh_dir = "DataPreparation/data"

emotion_mapping = {
    'A': 'Angry',
    'D': 'Disgust',
    'F': 'Fearful',
    'H': 'Happy',
    'N': 'Neutral',
    'SA': 'Sad',
    'SU': 'Surprise',
}

features = []
labels = []

for filename in os.listdir(bvh_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(bvh_dir, filename)
        df = pd.read_csv(file_path)
        print(f"Shape of DataFrame '{filename}': {df.shape}")

        features.append(df.values)

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

        labels.append(emotion)

print(features)
print(labels)
