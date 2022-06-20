import os
import re
import html
import librosa
import logging
import random
import torch
import numpy as np
import pickle5 as pickle
import pandas as pd
import pickle
from model import load_bert
from torchaudio.transforms import MFCC
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead

# LABEL_DICT={
#     '감정없음':0,
#     '놀람':1,
#     '슬픔':2,
#     '기쁨':3,
#     '분노':4,
#     '평온함(신뢰)':5,
#     '불안':6
# }

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(100)

LABEL_DICT={
    'angry':0,
    'neutral':1,
    'fear':2,
    'contempt':3,
    'sad':4,
    'surprise':5,
    'dislike':6,
    'happy':7
} #angry,neutral,fear,contempt,sad,surprise,dislike,happy

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def get_data_loader(args,
                    data_path,
                    #data_path2, #data_path3, data_path4, data_path5, data_path6, data_path7, data_path8,
                    bert_path,
                    num_workers,
                    batch_size,
                    split='train'):
    logging.info(f"loading {split} dataset")

    # paths
    data_path = os.path.join(data_path, f'{split}.pkl')
    vocab_path = os.path.join(bert_path, 'vocab.list')
    bert_args_path = os.path.join(bert_path, 'args.bin') #origin


    # MultimodalDataset object
    dataset = MultimodalDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        only_audio=args.only_audio,
        only_text=args.only_text
    )

    # collate_fn
    # batch sampler로 묶인 이후에는 collate_fn를 호출해 batch로 묶는다.
    collate_fn = AudioTextBatchFunction(
        args=args,
        pad_idx=dataset.pad_idx,
        cls_idx=dataset.cls_idx,
        sep_idx=dataset.sep_idx,
        bert_args=torch.load(bert_args_path),
        device='cpu' #origin
        #device="cuda"
    )

    return DataLoader(
        dataset=dataset,
        shuffle=False if split == 'train' else False,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True if split == 'train' else False
    )

class MultimodalDataset(Dataset): #tokenization, encoding, padding등을 해주는 코드
    """ Adapted from original multimodal transformer code"""

    def __init__(self,
                 data_path,
                 #data_path2,
                 vocab_path,
                 only_audio=False,
                 only_text=False):
        super(MultimodalDataset, self).__init__()
        self.only_audio = only_audio
        self.only_text = only_text
        self.use_both = not (self.only_audio or self.only_text)

        self.audio, self.text, self.labels = self.load_data(data_path)
        #audio2, text2, labels2 = self.load_data(data_path2)
        #self.audio2, self.text2, self.labels2 = self.load_data(data_path2)

        self.tokenizer, self.vocab = self.load_vocab(vocab_path)

        #self.text = self.text + text2 #+ text3 + text4 + text5 + text6 + text7 + text8
        #self.audio = self.audio + audio2 #+ audio3 + audio4 + audio5 + audio6 + audio7 + audio8
        #self.labels = self.labels + labels2 #+ labels3 + labels4 + labels5 + labels6 + labels7 + labels8
        self.tokenizer, self.vocab = self.load_vocab(vocab_path)

        # special tokens
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token_ids = None
        if not self.only_audio:
            tokens  = self.normalize_string(self.text.iloc[idx])
            tokens = self.tokenize(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # ------------------------guideline------------------------------------
        # naming as labels -> use to sampler
        # float32 is required for mfcc function in torchaudio
        # ---------------------------------------------------------------------
        #return self.audio[idx].astype(np.float32), token_ids, self.labels[idx]
        audio_arr=np.array(self.audio[idx]).astype(np.float32)
        label=self.labels[idx]
        return audio_arr,token_ids,label

        #return np.array(self.audio.iloc[#idx]).astype(np.float32), token_ids, self.labels[idx]

    def tokenize(self, tokens):
        return self.tokenizer.tokenize(tokens)

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    @staticmethod
    def load_data(path):
        #c1 = pd.read_pickle(path) #origin
        c1=open(path,'rb')
        data=pickle.load(c1)
        #data=data.reset_index() #
        #text = data['sentence'] #origin
        text=list(data['text_script'])
        #audio = data['audio'] #origin
        audio=list(data['audio'])
        #label = [LABEL_DICT[e] for e in data['emotion']] #origin
        label=[LABEL_DICT[e] for e in data['emotion']]
        del data
        return audio, text, label

    @staticmethod
    def load_vocab(path):
        #tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False) #origin
        tokenizer=AutoTokenizer.from_pretrained("beomi/kcbert-base")
        return tokenizer, tokenizer.vocab


class AudioTextBatchFunction:
    def __init__(self,
                 args,
                 pad_idx,
                 cls_idx,
                 sep_idx,
                 bert_args,
                 device='cpu'):
        self.device = device
        self.only_audio = args.only_audio
        self.only_text = args.only_text
        self.use_both = not (self.only_audio or self.only_text)

        # audio properties
        self.max_len_audio = args.max_len_audio
        self.n_mfcc = args.n_mfcc
        self.n_fft_size = args.n_fft_size
        self.sample_rate = args.sample_rate
        self.resample_rate = args.resample_rate

        # text properties
        self.max_len_bert = 300
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx

        # audio feature extractor #MFCC
        if not self.only_text:
            self.audio2mfcc = MFCC(
                sample_rate=self.resample_rate,
                n_mfcc=self.n_mfcc,
                log_mels=False,
                melkwargs={'n_fft': self.n_fft_size}
            ).to(self.device)

        # text feature extractor
        if not self.only_audio:
            self.bert = load_bert(args.bert_path, self.device)
            self.bert.eval()
            self.bert.zero_grad()

    def __call__(self, batch):
        audios, sentences, labels = list(zip(*batch))
        audio_emb, audio_mask, text_emb, text_mask = None, None, None, None
        with torch.no_grad():

            if not self.only_audio:
                #max_len = min(self.max_len_bert, max([len(sent) for sent in sentences]))
                global text_masks  # 추가
                max_len = self.max_len_bert
                input_ids = torch.tensor([self.pad_with_text(sent, max_len) for sent in sentences])
                text_masks = torch.ones_like(input_ids).masked_fill(input_ids == self.pad_idx, 0).bool()
                text_emb = self.bert(input_ids, text_masks)['last_hidden_state']

            if not self.only_text:
                audio_emb, audio_mask = self.pad_with_mfcc(audios)

        return audio_emb, audio_mask, text_emb, ~text_masks, torch.tensor(labels)

    def _add_special_tokens(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def pad_with_text(self, sentence, max_len):
        sentence = self._add_special_tokens(sentence)
        diff = max_len - len(sentence)
        if diff > 0:
            sentence += [self.pad_idx] * diff
        else:
            sentence = sentence[:max_len - 1] + [self.sep_idx]
        return sentence

    @staticmethod
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

    def pad_with_mfcc(self, audios):
        #max_len = min(self.max_len_audio, max([len(audio) for audio in audios]))
        max_len = self.max_len_audio
        audio_array = torch.zeros(len(audios), self.n_mfcc, max_len).fill_(float('-inf'))
        for idx, audio in enumerate(audios):
            # resample and extract mfcc
            audio = librosa.core.resample(audio, self.sample_rate, self.resample_rate)
            mfcc = self.audio2mfcc(torch.tensor(self._trim(audio)).to(self.device))

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
