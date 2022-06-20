import argparse
import logging
from tqdm import tqdm
from dataset import LABEL_DICT, get_data_loader
from model import MultimodalTransformer
import numpy as np
import pickle5 as pickle
import random
import os
import torch
import torch.nn as nn


def predict(model, data_loader, device):
    result=[]
    y_true=[]
    y_pred = []
    model.eval() #평가 상태로 지정
    model.zero_grad()

    iterator = tqdm(enumerate(data_loader), desc='predict_steps', total=len(data_loader))
    #with torch.no_grad():
    for step,batch in iterator:
        with torch.no_grad():
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
    model.load_state_dict(torch.load(args.model_path))

    #prediction
    logging.info('prediction starts')
    model.zero_grad()
    predict(model, data_loader, args.device)

    #model.zero_grad()
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
    #parser.add_argument('--data_path', type=str, default='C:/Users/hyeyoung/PycharmProjects/test/mult2/korean_audiotext_transformer/data')
    parser.add_argument('--data_path', type=str, default='E:/iitp/data_sep3')
    parser.add_argument('--bert_path', type=str, default='C:/Users/hyeyoung/PycharmProjects/test/mult2/korean_audiotext_transformer/KoBERT')
    #parser.add_argument('--model_path', type=str, default='E:/[안전폴더]epoch1-loss1.7936-f10.2050/epoch1-loss1.5609-f10.3704.pt')
    parser.add_argument('--model_path', type=str,default='KoBERT/epoch1-loss1.5609-f10.3704.pt')
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
