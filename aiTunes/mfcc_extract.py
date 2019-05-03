import pandas as pd
import numpy as np
import warnings
import librosa
from os import listdir
from os.path import isfile, join
from keras.models import load_model
import csv

warnings.filterwarnings("ignore")
AUDIO_DIR = 'songs/'
PARENT_DIR = ''


def extract_feature(path):
    # Traversing over each file in path
    for f in listdir(path):
        song_id = f.replace('.mp3', '')
        print(f)
        mfcc_vector = pd.Series()
        # Reading Song
        songname = path + f

        for i in range(0, 45, 2):
            y, sr = librosa.load(songname, duration=2, offset=i)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
            mfcc_avg = np.mean(mfcc, axis=1)
            mfcc_vector.set_value(i, mfcc_avg)  # song name
        mfcc_vector.to_csv(f'mfcc/{song_id}.csv')


extract_feature(AUDIO_DIR)