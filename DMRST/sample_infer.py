import os
import torch
import numpy as np
import argparse
import os
import config
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet
import pandas as pd
from nltk.tokenize import sent_tokenize

#os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    print(input_sentences)
    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred


if __name__ == '__main__':

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    print("Loaded Model")
    
    df = pd.read_csv(args.data_path)
    print(len(df))
    docs = list(df['source'])
    summaries = list(df['summary'])
    ## Infer on summary

    summ_sents = []
    summ_segments = []
    summ_tree_parsing = []
    for summ in summaries:
        Test_InputSentences = summ
    #Test_InputSentences = open("./data/text_for_inference copy.txt").readlines()

        try:
            input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(model, bert_tokenizer, Test_InputSentences, batch_size)
            summ_sents.append( input_sentences)
            summ_segments.append(all_segmentation_pred)
            summ_tree_parsing.append(all_tree_parsing_pred)
            print(all_segmentation_pred[0])
        except:

            summ_sents.append(sent_tokenize(summ))
            summ_segments.append([])
            summ_tree_parsing.append([])
            continue
        #output.to_csv("outputs/%s_out.csv"%fname)
    df['summ_sents'] = summ_sents
    df['summ_segments']= summ_segments
    df['summ_tree_parsing'] = summ_tree_parsing
    df.to_csv(args.data_path.replace(".csv", "_parsed.csv"))