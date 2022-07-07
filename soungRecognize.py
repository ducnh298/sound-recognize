import os
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
import pickle
#from annoy import AnnoyIndex
from collections import Counter

def extract_features(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")

def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    print(len(crop_feat))
    return crop_feat


#Trích rút đặc trưng toàn bộ file data
data_dir='E:\Document\Tai Lieu\He CSDL Da Phuong Tien\Data'
features = []
songs = []


for song in tqdm(os.listdir(data_dir)):
    song = os.path.join(data_dir, song)
    y, sr = librosa.load(song, sr=16000)
    feat = extract_features(y)
    for i in range(0, feat.shape[0] - 10, 5):
        feat100 = list(crop_feature(feat, i, nb_step=10))
        features.append(feat100)
        songs.append(song)


pickle.dump(features, open('features.pk', 'wb'))
pickle.dump(songs, open('songs.txt', 'wb'))

f = 100
t = AnnoyIndex(f)

for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)

t.build(100) 
t.save('annoydata.txt')


# #load dữ liệu
# features = np.load('E:\Document\Tai Lieu\He CSDL Da Phuong Tien\Du lieu code python\features.pk', allow_pickle=True)
# songs = np.load('E:\Document\Tai Lieu\He CSDL Da Phuong Tien\Du lieu code python\songs.pk', allow_pickle=True)

f = 100
u = AnnoyIndex(f)
u.load('annoydata.txt')


#Test one file
song = os.path.join(data_dir, 'E:\Document\Tai Lieu\He CSDL Da Phuong Tien\Du lieu code python\data test\Piano 60s performance - Pianist in tears!!!..wav')
y, sr = librosa.load(song, sr=16000)
feat = extract_features(y)

results = []
for i in range(0, feat.shape[0], 10):
    new_feat = list(crop_feature(feat, i, nb_step=10))
    result = u.get_nns_by_vector(new_feat, 5)
    result_songs = [songs[k] for k in result]
    results.append(result_songs)

results = np.array(results).flatten()
most_song = Counter(results)
print("\n\n", song[75:]," is recognized as ")
for result_song in most_song.most_common(5):
  print("\t\t",result_song[0][69:],",",result_song[1])

##Test all data
# test_data_dir='E:\Document\Tai Lieu\He CSDL Da Phuong Tien\Data test'

# for song in tqdm(os.listdir(test_data_dir)):
#     song_name =song
#     song = os.path.join(test_data_dir, song)
#     y, sr = librosa.load(song, sr=16000)
#     zcrs = sum(librosa.zero_crossings(y))
#     feat = extract_features(y)
#     results = []
#     for i in range(0, feat.shape[0], 10):
#         new_feat = list(crop_feature(feat, i, nb_step=10))
#         new_feat.append(zcrs)
#         result = u.get_nns_by_vector(new_feat, 5)
#         result_songs = [songs[k] for k in result]
#         results.append(result_songs)

#     results = np.array(results).flatten()
#     most_song = Counter(results)
#     print("\n\n", song_name," recognize as ")
#     for result_song in most_song.most_common(5):
#         print("\t\t",result_song[0][69:],",",result_song[1])
#     print("\n")