"""
Modified from TalkNet / LR-ASD (https://github.com/Junhua-Liao/LR-ASD)

AVA-ActiveSpeaker data loaders.

Each line of a ``{split}_loader.csv`` file describes one face track
(tab-separated)::

    <entity_id>\t<num_frames>\t<fps>\t[<label_0>,...,<label_{T-1}>]\t<start_index>

``entity_id`` starts with the 11-character YouTube video id. The face crops
live in ``clips_videos/{split}/<video_id>/<entity_id>/<timestamp>.jpg`` and the
audio in ``clips_audios/{split}/<video_id>/<entity_id>.wav`` (16 kHz). Audio is
turned into 13-dim MFCCs with 4 audio frames per video frame.
"""

import os
import glob
import random

import cv2
import numpy
import torch
import python_speech_features
from scipy.io import wavfile


def generate_audio_set(dataPath, batchList):
    """Load the raw waveforms of all tracks in a mini-batch.

    Args:
        dataPath (str): Root folder of the audio clips (``clips_audios/{split}``).
        batchList (list[str]): CSV lines of the tracks in the mini-batch.

    Returns:
        dict[str, numpy.ndarray]: Map from ``entity_id`` to its int16 waveform.
    """
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):
    """Audio augmentation: mix in another track's audio at a random SNR in [-5, 5] dB.

    Args:
        dataName (str): ``entity_id`` of the current track (excluded as the noise source).
        audio (numpy.ndarray): Waveform of the current track.
        audioSet (dict[str, numpy.ndarray]): Waveforms of the mini-batch (the noise
            candidates). Must hold at least two tracks.

    Returns:
        numpy.ndarray: Mixed int16 waveform with the same length as ``audio``.
    """
    remaining = list(set(list(audioSet.keys())) - {dataName})
    noiseName =  random.sample(remaining, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    """Compute the MFCC features of a track, aligned to ``numFrames`` video frames.

    The MFCC window and step are scaled by ``25 / fps``, so there are always 4
    audio frames per video frame. Short sequences are wrap-padded.

    Args:
        data (list[str]): Split CSV line of the track.
        dataPath (str): Unused (the waveform comes from ``audioSet``).
        numFrames (int): Number of video frames to align to.
        audioAug (bool): If True, apply :func:`overlap` with probability 0.5.
        audioSet (dict[str, numpy.ndarray]): Preloaded waveforms (see
            :func:`generate_audio_set`).

    Returns:
        numpy.ndarray: MFCCs of shape (4 * numFrames, 13).
    """
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(data, dataPath, numFrames, visualAug):
    """Load the grayscale 112x112 face crops of a track.

    Args:
        data (list[str]): Split CSV line of the track.
        dataPath (str): Root folder of the face crops (``clips_videos/{split}``).
        numFrames (int): Number of frames to load.
        visualAug (bool): If True, apply one augmentation to the whole track,
            chosen at random from ``orig``, horizontal ``flip``, random ``crop``
            (70-100% of the side, resized back) and ``rotate`` (+-15 degrees).

    Returns:
        numpy.ndarray: uint8 faces of shape (numFrames, 112, 112).
    """
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.stack(faces)
    return faces


def load_label(data, numFrames):
    """Parse the per-frame speaking labels of a track.

    Args:
        data (list[str]): Split CSV line of the track.
        numFrames (int): Number of labels to keep.

    Returns:
        numpy.ndarray: int labels (0/1) of shape (numFrames,).
    """
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.stack(res[:numFrames])
    return res


class train_loader(object):
    """AVA training dataset whose items are whole mini-batches of similar-length tracks.

    Tracks are sorted by length (longest first) and grouped so that each
    mini-batch holds about ``batchSize`` frames in total. Every track in a group
    is cut to the length of the shortest one. Use it with a ``DataLoader`` with
    ``batch_size=1``.

    Args:
        trialFileName (str): Path to ``csv/train_loader.csv``.
        audioPath (str): Path to ``clips_audios/train``.
        visualPath (str): Path to ``clips_videos/train``.
        batchSize (int): Maximum number of frames per mini-batch.
    """

    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0    
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end

    def __getitem__(self, index):
        """Load mini-batch ``index`` with audio and visual augmentation.

        Returns:
            dict: ``'audio'`` (N, 13, 4*T) float, ``'video'`` (N, T, 1, 112, 112)
            float in [0, 1], and ``'labels'`` (N, T) long, where N is the number
            of tracks in the mini-batch.
        """
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.visualPath, numFrames, visualAug = True))
            labels.append(load_label(data, numFrames))
        return {
            'audio': torch.FloatTensor(numpy.stack(audioFeatures)).transpose(-1, -2), 
            'video': torch.FloatTensor(numpy.stack(visualFeatures)).unsqueeze(2) / 255, 
            'labels': torch.LongTensor(numpy.stack(labels)) 
            }       

    def __len__(self):
        """Return the number of mini-batches."""
        return len(self.miniBatch)


class val_loader(object):
    """AVA evaluation dataset: one full face track per item, no augmentation.

    Args:
        trialFileName (str): Path to ``csv/{split}_loader.csv``.
        audioPath (str): Path to ``clips_audios/{split}``.
        visualPath (str): Path to ``clips_videos/{split}``.
    """

    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        """Load track ``index``.

        Returns:
            dict: ``'audio'`` (1, 13, 4*T), ``'video'`` (1, T, 1, 112, 112) and
            ``'labels'`` (1, T).
        """
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return {
            'audio': torch.FloatTensor(numpy.array(audioFeatures)).transpose(-1, -2), 
            'video': torch.FloatTensor(numpy.array(visualFeatures)).unsqueeze(2) / 255, 
            'labels': torch.LongTensor(numpy.array(labels)) 
            }  

    def __len__(self):
        """Return the number of tracks."""
        return len(self.miniBatch)
