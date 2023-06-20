from flask import Flask, request
import nemo.collections.asr as nemo_asr
import numpy as np

from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from omegaconf import OmegaConf
import copy
from torch.utils.data import DataLoader
import torch

asr_model = nemo_asr.models.ASRModel.restore_from('stt_ru_conformer_ctc_large.nemo')
cfg = copy.deepcopy(asr_model._cfg)
# sample rate, Hz
SAMPLE_RATE = 16000
OmegaConf.set_struct(cfg.preprocessor, False)


asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
# Set model to inference mode
asr_model.eval()
asr_model = asr_model.to(asr_model.device)

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
    return predictions


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
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')

        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()

    def _decode(self, frame, offset=0):
        print(len(frame), self.n_frame_len)
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(asr_model, self.buffer).cpu().numpy()[0]
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap],
            self.vocab
        )
        return decoded[:len(decoded) - offset]


@torch.no_grad()
# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(
        asr_model.device)
    log_probs, encoded_len, predictions = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return predictions, encoded_len


def reset(self):
    '''
    Reset frame_history and decoder's state
    '''
    self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
    self.prev_char = ''


vocab = list(cfg.decoder.vocabulary)
vocab.append('_')
asr_model.eval()
asr_model.encoder.freeze()
asr_model.decoder.freeze()

app = Flask(__name__)


@app.route('/nemo', methods=['POST'])


def launch_task():
    if request.method == 'POST':

        if 'data' not in request.files:
            print('no files found')


        #читаем данные в однобайтовый объект
        data = request.files.get('data')
        data = data.read()

        signal = np.frombuffer(data, dtype=np.int16) / 32768.

        with torch.no_grad():
            logits, logits_len = infer_signal(asr_model, signal)
            current_hypotheses, all_hyp = asr_model.decoding.ctc_decoder_predictions_tensor(
                logits, decoder_lengths=logits_len
            )

        result = current_hypotheses[0]

        return result


@app.route('/')


def index():
    return('Nemo inference')


if __name__ == '__main__':
    app.run(debug=False, port=4999)
