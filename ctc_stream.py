import nemo.collections.asr as nemo_asr
import numpy as np
import pyaudio as pa
import time
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from omegaconf import OmegaConf
import copy
from torch.utils.data import DataLoader

asr_model = nemo_asr.models.ASRModel.restore_from('stt_ru_conformer_ctc_large.nemo')
cfg = copy.deepcopy(asr_model._cfg)

SAMPLE_RATE = 16000
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0

asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
# Set model to inference mode
asr_model.eval()

asr_model = asr_model.to(asr_model.device)

asr_model.encoder.freeze()
asr_model.decoder.freeze()

# Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)


class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)


# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(
        asr_model.device)
    log_probs, encoded_len, predictions = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return predictions, encoded_len


# %%
# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:

    def __init__(self, model_definition,
                 frame_len=0.5, frame_overlap=1,
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(cfg.decoder.vocabulary)
        self.vocab.append('_')

        self.sr = 16000
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']

        # timestep_duration *= 1  # block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits, logits_len = infer_signal(asr_model, self.buffer)
        current_hypotheses, all_hyp = asr_model.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len
        )
        return current_hypotheses

    @torch.no_grad()
    def transcribe(self, frame=None, merge=False):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return unmerged

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    # @staticmethod
    # def _greedy_decoder(logits, vocab):
    #     s = ''
    #     for i in range(logits.shape[0]):
    #         s += vocab[np.argmax(logits[i])]
    #     return s

    # def greedy_merge(self, s):
    #     s_merged = ''
    #
    #     for i in range(len(s)):
    #         if s[i] != self.prev_char:
    #             self.prev_char = s[i]
    #             if self.prev_char != '_':
    #                 s_merged += self.prev_char
    #     return s_merged


# %%
# duration of signal frame, seconds
FRAME_LEN = 5
# number of audio channels (expect mono signal)
CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN * SAMPLE_RATE)
asr = FrameASR(model_definition={
    'sample_rate': SAMPLE_RATE,
    'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
    'ConformerEncoder': cfg.encoder,
    'labels': cfg.decoder.vocabulary
},
    frame_len=FRAME_LEN, frame_overlap=1,
    offset=0)
# %%
asr.reset()

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    empty_counter = 0


    def callback(in_data, frame_count, time_info, status):
        global empty_counter
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = asr.transcribe(signal)
        if len(text):
            print(text, end=' ')
            empty_counter = asr.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ', end='')
        return in_data, pa.paContinue


    stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK_SIZE)

    print('Listening...')

    stream.start_stream()

    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")

else:
    print('ERROR: No audio input device found.')
