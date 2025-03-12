#This script contains key functions for loading and transforming audio data.

#IMPORTS
import librosa, os #audio processing and file system parsing
import numpy as np #math library
import tensorflow as tf #for model building
import pandas as pd #for data analysis / prep

# Parameters
SAMPLE_RATE = 22050  # Standard sample rate for music processing
N_MELS = 128         # Number of Mel filterbanks
HOP_LENGTH = 512     # Hop length for STFT
N_FFT = 2048         # FFT window size
DURATION = 5         # Duration of each audio clip in seconds
BATCH_SIZE = 32

def load_audio_to_mel(file_path):
    """Function to load an audio file and convert it to a Mel spectrogram."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to log scale (dB)
    
    # Normalize to [0,1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm

def mel_to_audio(mel_spec, sr=22050, n_fft=2048, hop_length=512, n_mels=128, power=1.0):
    """
    Convert a Mel spectrogram back to audio using the Griffin-Lim algorithm.
    Args:
        mel_spec: Mel spectrogram (shape: [n_mels, time_steps])
        sr: Sample rate for the audio
        n_fft: FFT size for Griffin-Lim
        hop_length: Hop length for Griffin-Lim
        n_mels: Number of Mel bins in the spectrogram
        power: Exponent for the spectrogram
    Returns:
        Audio signal as a numpy array
    """
    # Invert Mel to linear scale
    mel_inverted = librosa.feature.inverse.mel_to_audio(mel_spec ** power, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return mel_inverted

def load_music_df(music_dir: str):
    """This function loads a directory of songs and creates a simple dataframe with (genre,song,numpy) rows. We expect music_dir to be a 2-level directory with genre directories on the first level and .wav songs on the seocnd level."""
    music_dicts,bad_paths = [],[]
    genres = os.listdir(music_dir)
    for genre in genres:
        try:
            for song in os.listdir(music_dir+genre):
                song_path = music_dir+genre+'/'+song
                try:
                    music_dicts.append({'genre': genre, 'song': song, 'numpy_representation': load_audio_to_mel(song_path)})
                except Exception as e:
                    print(f"couldn't load: {song_path}, got: {e}")
                    bad_paths.append([song_path,e])
        except Exception as e:
            print(f"couldn't process the {genre} directory, got: {e}")
    return pd.DataFrame(music_dicts)

def create_tf_dataset(music_df):
    """This convenience function converts our music_df's numpy column to a tensorflow-ready dataset."""
    numpy_representations = np.array(music_df["numpy_representation"].tolist(), dtype=np.float32)  
    numpy_representations = np.expand_dims(numpy_representations, -1)  # Add channel dimension
    songs_dataset = tf.data.Dataset.from_tensor_slices(numpy_representations)
    songs_dataset = songs_dataset.shuffle(len(numpy_representations)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return songs_dataset