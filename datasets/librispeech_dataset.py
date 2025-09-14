import numpy as np
import os
from torch.utils.data import Dataset

import soundfile
import webrtcvad


class LibriSpeechDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def __init__(self, path, T, size=None, return_vad=False, readers_range=None):
		self.corpus = self._exploreCorpus(path, 'flac')
		if readers_range is not None:
			for key in list(map(int, self.nChapters.keys())):
				if int(key) < readers_range[0] or int(key) > readers_range[1]:
					del self.corpus[key]

		self.nReaders = len(self.corpus)
		self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
		self.nUtterances = {reader: {
				chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
			} for reader in self.corpus.keys()}

		self.chapterList = []
		for chapters in list(self.corpus.values()):
			self.chapterList += list(chapters.values())

		self.fs = 16000
		self.T = T

		self.return_vad = return_vad
		self.vad = webrtcvad.Vad()

		self.sz = len(self.chapterList) if size is None else size

	def _exploreCorpus(self, path, file_extension):
		directory_tree = {}
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
		return directory_tree

	def _cleanSilences(self, s, aggressiveness, return_vad=False):
		self.vad.set_mode(aggressiveness)

		vad_out = np.zeros_like(s)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(s) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
		s_clean = s * vad_out

		return (s_clean, vad_out) if return_vad else s_clean

	def __len__(self):
		return self.sz
   
	def __getitem__(self, idx):
		# Generate random length between 2 and max_T
		random_len = np.random.randint(2, self.T + 1)  # +1 because upper bound is exclusive

		if idx < 0: idx = len(self) + idx
		while idx >= len(self.chapterList): idx -= len(self.chapterList)
		chapter = self.chapterList[idx]

		# Get a random speech segment from the selected chapter
		s = np.array([])
		utt_paths = list(chapter.values())
		n = np.random.randint(0, len(chapter))

		# Collect enough audio for the random length (not max_T)
		while s.shape[0] < random_len * self.fs:
			utterance, fs = soundfile.read(utt_paths[n])
			assert fs == self.fs
			s = np.concatenate([s, utterance])
			n += 1
			if n >= len(chapter): n = 0

		# Trim to exactly random_len seconds
		s = s[0: random_len * self.fs]
		s -= s.mean()

		# Clean silences
		s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

		# Pad with zeros to reach max_T length
		if len(s_clean) < self.T * self.fs:
			pad_length = self.T * self.fs - len(s_clean)
			s_clean = np.pad(s_clean, (0, pad_length), mode='constant')
			if self.return_vad:
				vad_out = np.pad(vad_out, (0, pad_length), mode='constant')

		return (s_clean, vad_out) if self.return_vad else s_clean

