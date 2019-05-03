from flask import Flask
import pandas as pd
from flask import render_template, send_from_directory, send_file, request
import pandas as pd
import json
import numpy as np
import warnings
import librosa
from os import listdir
from os.path import isfile, join
from keras.models import load_model
import csv
from collections import deque
from difflib import SequenceMatcher

app = Flask(__name__)

session_count = 0
last_ten_ids = deque()
prev_emotions = []

@app.route('/')
def hello_world():
    df = pd.read_csv("Song_Meta.csv")
    return render_template('index.html', form=parse_vals(df.values))


@app.route('/songs/<path:filename>')
def download_file(filename):
    return send_file(f"./songs/{filename}")


@app.route('/queue', methods=['POST'])
def queue():
    global session_count, last_ten_ids
    session_count += 1
    jsonData = request.get_json()
    print(jsonData)
    song_id = int(jsonData['song_id'])
    if song_id not in last_ten_ids:
        if len(last_ten_ids) == 10:
            last_ten_ids.pop()
        last_ten_ids.appendleft(song_id)
    metadata_sim = np.array(find_metadata_similarity(song_id))
    emotion_sim = np.array(find_emotion_similarity(song_id))
    MAX_EMOTION = max(emotion_sim[:, 1])
    MIN_EMOTION = min(emotion_sim[:, 1])
    emotion_sim[:, 1] = (emotion_sim[:, 1] - MIN_EMOTION) / (MAX_EMOTION - MIN_EMOTION)
    MAX_METSCORE = max(metadata_sim[:, 1])
    MIN_METSCORE = min(metadata_sim[:, 1])
    metadata_sim[:, 1] = (metadata_sim[:, 1] - MIN_METSCORE) / (MAX_METSCORE - MIN_METSCORE)
    combined = {}
    for song in song_ids:
        if song == song_id: continue
        metadata_score = metadata_sim[metadata_sim[:, 0] == song][0, 1]
        emotion_score = emotion_sim[emotion_sim[:, 0] == song][0, 1]
        combined_score = 0.75 * (1 - emotion_score) + 0.25 * metadata_score - 1 * (song in last_ten_ids)
        combined[int(song)] = combined_score
    queue = []
    queue_temp = np.array(sorted(combined.items(), key=lambda x: x[1], reverse=True))[:10, 0].tolist()
    df = pd.read_csv("Song_Meta.csv", index_col='id')
#     print(df)
    for song in queue_temp:
        queue.append([f"songs/{int(song)}.mp3", df.loc[song].values[0]])
    return json.dumps(queue)


def parse_vals(meta):
    output = []
    queue = []
    for row in meta:
        output.append([f"songs/{row[0]}.mp3", row[1]])
    return [output, queue]


SONGS_DIR = 'songs/'

print("Predicting emotions")


def predict(path):
    model = load_model('mfcc_trained.hd5')
    # Traversing over each file in path
    output_vector = {}
    for f in listdir(path):
        if '.mp3' not in f: continue
        print(f)
        mfcc_vector = []
        # Reading Song
        songname = path + f
        for i in range(0, 45, 2):
            y, sr = librosa.load(songname, duration=2, offset=i)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
            mfcc_avg = np.mean(mfcc, axis=1)
            mfcc_vector.append(mfcc_avg)  # song name
        mfcc_vector = np.array(mfcc_vector)
        mfcc_vector = np.reshape(mfcc_vector, (1, mfcc_vector.shape[0], mfcc_vector.shape[1]))
        output_vector[f.replace(".mp3", '')] = model.predict(mfcc_vector)
    return output_vector


emotions = predict(SONGS_DIR)

print("Finished predicting emotions")

print("Creating metadata dictionary")


def createdic():
    dic = {}
    df = pd.read_csv("Song_Meta.csv", index_col='id')
    print(df.shape)
    song_ids = df.index.tolist()
    print(len(song_ids))
    for song_id in song_ids:
        dic[song_id] = df.loc[song_id].values[1] + " " + df.loc[song_id].values[2]
    return song_ids, dic


song_ids, song_metadic = createdic()


def find_metadata_similarity(song_id):
    metascore = []
    for key in song_metadic.keys():
        if key == song_id:
            continue
        metascore.append([key, SequenceMatcher(None, song_metadic[key], song_metadic[song_id]).ratio()])
    return metascore


# Find Emotion Similarity
def find_emotion_similarity(song_id):
    this_song = np.array([emotions[str(song_id)][0][0], emotions[str(song_id)][0][1]])
    global prev_emotions, session_count
    prev_emotions.append([emotions[str(song_id)][0][0], emotions[str(song_id)][0][1]])
    if session_count >= 5:
        this_song = interpolate(prev_emotions)
    sim_scores = []
    for song in song_ids:
        if song_id == song: continue
        curr_song = np.array([emotions[str(song)][0][0], emotions[str(song)][0][1]])
        dist = np.linalg.norm(this_song - curr_song)
        sim_scores.append([song, dist])
    return sim_scores


AROUSAL_MIN, AROUSAL_MAX = 1, 9
VALENCE_MIN, VALENCE_MAX = 1, 9
DEG = 5


def emotion_polyfit(arousal_means, valence_means):
    l = len(arousal_means)
    arousal_predict = np.poly1d(np.polyfit(range(1, l+1), arousal_means, 3))
    valence_predict = np.poly1d(np.polyfit(range(1, l+1), valence_means, 3))

    arousal_val = arousal_predict(l+1)
    valence_val = valence_predict(l+1)

    # print(f'arousal predict: {arousal_val}, valence predict: {valence_val}')

    arousal_means = np.append(arousal_means, arousal_val)
    valence_means = np.append(valence_means, valence_val)
    return arousal_means, valence_means


def interpolate(data, num_times=1):
    data = np.array(data).T
    arousal_mean, valence_mean = data[0], data[1]
    for _ in range(num_times):
        arousal_mean, valence_mean = emotion_polyfit(arousal_mean, valence_mean)
    for i in range(1):
        arousal_mean[~i] = max(AROUSAL_MIN, min(AROUSAL_MAX, arousal_mean[~i]))
        valence_mean[~i] = max(VALENCE_MIN, min(VALENCE_MAX, valence_mean[~i]))

    return np.array([arousal_mean[-1], valence_mean[-1]])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50000)
