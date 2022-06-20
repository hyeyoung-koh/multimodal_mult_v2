import argparse
import logging
from tqdm import tqdm
from dataset import LABEL_DICT, get_data_loader
from model import MultimodalTransformer
import numpy as np
import random
import os
import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
bert_path = './KoBERT'
from KoBERT_model.tokenization import BertTokenizer
from torchaudio.transforms import MFCC
import os
from transformers import BertConfig, BertModel
import json


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def clean_state_dict(state_dict):
    new = {}
    for key, value in state_dict.items():
        if key in ['fc.weight', 'fc.bias']:
            continue
        new[key.replace('bert.', '')] = value
    return new

def load_bert(bert_path):
    bert_config_path = os.path.join(bert_path, 'config.json')
    bert = BertModel(BertConfig(vocab_size=30797, **load_json(bert_config_path))).cuda()
    bert_model_path = os.path.join(bert_path, 'model.bin')
    bert.load_state_dict(clean_state_dict(torch.load(bert_model_path)), strict=False)
    return bert

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(50)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

rng = np.random.default_rng(8888)
rfloat = rng.random()

def predict(model, data_loader, device):
    result=[]
    y_true=[]
    y_pred = []
    model.eval() #평가 상태로 지정
    bert=load_bert(bert_path)
    bert.eval()
    bert.zero_grad()

    iterator = tqdm(enumerate(data_loader), desc='predict_steps', total=len(data_loader))
    with torch.no_grad():
        for step,batch in iterator:
            batch = map(lambda x: x.to(device) if x is not None else x, batch)
            audios, a_mask, texts, t_mask, labels = batch
            labels = labels.squeeze(-1).long() #labels
            logit,hidden=model(audios,texts,a_mask,t_mask)
            softmax_layer = nn.Softmax(-1)
            softmax_result = softmax_layer(logit)
            print('logit:',logit)
            print('predict_result:',y_pred)
            print('prob:',softmax_result)
            print('labels:',labels)
            y_true.append(labels)
            y_pred.append(logit.max(dim=1)[1])
    print('y_pred:',torch.flatten(torch.tensor(y_pred)))
    y_pred_list=torch.flatten(torch.tensor(y_pred))
    print('y_true:',torch.flatten(torch.tensor(y_true)))
    y_true_list=torch.flatten(torch.tensor(y_true))
    return y_pred_list,y_true_list

def main(args):
    data_loader = get_data_loader(
        args=args,
        data_path=args.data_path,
        bert_path=args.bert_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        split=args.split
    )

    model = MultimodalTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=args.n_classes,
        only_audio=args.only_audio,
        only_text=args.only_text,
        d_audio_orig=args.n_mfcc,
        d_text_orig=768,  # BERT hidden size
        d_model=args.d_model,
        attn_mask=args.attn_mask
    ).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    #prediction
    logging.info('prediction starts')


    model.zero_grad()
    import pandas as pd
    final_pred,final_true=predict(model, data_loader, args.device)
    df=pd.read_csv('E:/predict_list0509.csv')
    df['prediction']=final_pred
    df['true_value']=final_true
    df.to_csv('E:/new_predict_list0509.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--split', type=str, default='test') #test dataset
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='E:/iitp/data_sep3')
    parser.add_argument('--bert_path', type=str, default='C:/Users/hyeyoung/PycharmProjects/test/mult2/korean_audiotext_transformer/KoBERT')
    parser.add_argument('--model_path', type=str, default='E:/[안전폴더]epoch1-loss1.7936-f10.2050/epoch1-loss1.7936-f10.2050.pt')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=0) #8
    parser.add_argument('--batch_size', type=int, default=1) #8

    # architecture
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=40) #origin
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--attn_mask', action='store_false')

    # data processing
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=400)
    #parser.add_argument('--n_mfcc', type=int, default=40) #origin
    parser.add_argument('--n_mfcc', type=int, default=64)

    args_ = parser.parse_args()

    # check usage of modality
    if args_.only_audio and args_.only_text:
        raise ValueError("Please check your usage of modalities.")

    # seed and device setting
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #이게 원래
    args_.device = device_

    # log setting
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    main(args_)
