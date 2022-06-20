import torch
from pydub import AudioSegment
import config
import numpy as np
import librosa
import os
import html
import re
import json
from transformers import BertConfig, BertModel
import torch.nn as nn
from KoBERT.tokenization import BertTokenizer
from model import MultimodalTransformer

def extract_audio_array(wav_file):
    audio = AudioSegment.from_wav(wav_file)
    audio = audio.set_channels(1)
    audio = audio.get_array_of_samples()
    return np.array(audio).astype(np.float32)


def _trim(audio):
    left, right = None, None
    for idx in range(len(audio)):
        if np.float32(0) != np.float32(audio[idx]):
            left = idx
            break
    for idx in reversed(range(len(audio))):
        if np.float32(0) != np.float32(audio[idx]):
            right = idx
            break
    return audio[left:right + 1]


def pad_with_mfcc(wav_file):
    max_len = config.max_len_audio
    audio_array = torch.zeros(len([wav_file]), config.n_mfcc, max_len).fill_(float('-inf'))
    for idx, audio in enumerate([wav_file]):
        # resample and extract mfcc
        audio = librosa.core.resample(audio, config.sample_rate, config.resample_rate)
        mfcc = config.audio2mfcc(torch.tensor(_trim(audio)).cuda())

        # normalize
        cur_mean, cur_std = mfcc.mean(dim=0), mfcc.std(dim=0)
        mfcc = (mfcc - cur_mean) / cur_std

        # save the extracted mfcc
        cur_len = min(mfcc.shape[1], max_len)
        audio_array[idx, :, :cur_len] = mfcc[:, :cur_len]

    # (batch_size, n_mfcc, seq_len) -> (batch_size, seq_len, n_mfcc)
    padded = audio_array.transpose(2, 1)

    # get key mask
    key_mask = padded[:, :, 0]
    key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0)
    key_mask = key_mask.masked_fill(key_mask == float('-inf'), 1).bool()

    # -inf -> 0.0
    padded = padded.masked_fill(padded == float('-inf'), 0.)
    return padded, key_mask


def _add_special_tokens(token_ids):
    return [config.cls_idx] + token_ids + [config.sep_idx]


def pad_with_text(sentence, max_len):
    sentence = _add_special_tokens(sentence)
    diff = max_len - len(sentence)
    if diff > 0:
        sentence += [config.pad_idx] * diff
    else:
        sentence = sentence[:max_len - 1] + [config.sep_idx]
    return sentence


def clean_state_dict(state_dict):
    new = {}
    for key, value in state_dict.items():
        if key in ['fc.weight', 'fc.bias']:
            continue
        new[key.replace('bert.', '')] = value
    return new


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_bert(bert_path):
    bert_config_path = os.path.join(config.bert_path, 'config.json')
    bert = BertModel(BertConfig(vocab_size=30797, **load_json(bert_config_path))).cuda()
    bert_model_path = os.path.join(bert_path, 'model.bin')
    bert.load_state_dict(clean_state_dict(torch.load(bert_model_path)), strict=False)
    return bert


def load_vocab(path):
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    return tokenizer, tokenizer.vocab


def normalize_string(s):
    s = html.unescape(s)
    s = re.sub(r"[\s]", r" ", s)
    s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
    return s


def tokenize(tokens):
    return config.tokenizer.tokenize(tokens)
