'''

 특징을 추출해서 npz로 저장하는 코드입니다

사용하는 소리특징은 MFCCs, Chromagram , Melspectrogram, Spectral Contrast, Tonal centroid features(Tonnetz) 입니다.

 소리 파일은 어떤 한 부분이 소리의 특징이라고 할 수 없으므로,
 전체를 조각내어 특징을 추출한 후 평균낸 것을 소리의 특징이라고 하였습니다.

'''

import glob
import librosa
import numpy as np


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(filenames):
    rows = len(filenames)
    features, labels = np.zeros((rows,193)), np.zeros((rows,4))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('\\')[2].split('-')[0])
        except:
            print(fn)
        else:
            print("step :"+str(i+1) )
            features[i] = ext_features
            labels[i, y_col] = 1
            i += 1
    return features, labels


audio_files = []

audio_files.extend(glob.glob('Sound_Set\\realtime-sound-set\\*.wav'))

print(len(audio_files))

files = audio_files[0: len(audio_files)]

X, y = parse_audio_files(files)

for r in y:
    if np.sum(r) > 1.5:
        print('error occured')
        break

np.savez('Sound_set/realtime_sound_set', X=X, y=y)