import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import pandas as pd
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable

# from utils.print_args import print_args
from sklearn.metrics import  f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def classification(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

def get_args():
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet, iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, default='Ours data', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=300, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # classification task
    parser.add_argument('--pred_type', type=str, default='quality', help='options: [quality, ra, readiness]')
    parser.add_argument('--num_class', type=int, default=3, help='quality:3, ra:3, readiness:2')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=True, action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    return parser.parse_args()  

def get_data(pred_type):
    au_data = np.load('DATA/aus.npy').transpose(0, 2, 1)
    em_data = np.load('DATA/ems.npy').transpose(0, 2, 1)
    hp_data = np.load('DATA/hps.npy').transpose(0, 2, 1)

    bp_data = np.load('DATA/bps.npy')
    bp_data = bp_data.reshape(bp_data.shape[0], bp_data.shape[1], -1).transpose(0, 2, 1)

    combine_data =np.column_stack((au_data, em_data, hp_data, bp_data))    # person_num * seq_num * seq_len(300)

    quality_label = np.load('DATA/quality.npy', allow_pickle=True)
    quality_label = np.array(quality_label, dtype=int)
    ra_label = np.load('DATA/ra.npy', allow_pickle=True)
    ra_label = np.array(ra_label, dtype=int)
    readiness_label = np.load('DATA/readiness.npy', allow_pickle=True)
    readiness_label = np.array(readiness_label, dtype=int)

    if pred_type == 'quality':
        combine_label = quality_label
    elif pred_type == 'ra':
        combine_label = ra_label
    else:
        combine_label = readiness_label

    return combine_data, combine_label

if __name__ == '__main__':

    args = get_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print_args(args)

    if args.task_name == 'classification':
        
        combine_data, combine_label = get_data(args.pred_type)
        
        args.seq_len = combine_data.shape[1]
        args.enc_in = combine_data.shape[2]

        avg_acc = []
        avg_f1 = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(combine_data, combine_label):

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = Model(args).to(device=device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.95)
            criterion = nn.CrossEntropyLoss()

            X_train = torch.tensor(np.array(combine_data)[train_index], dtype=torch.float32)
            y_train = torch.tensor(np.array(combine_label)[train_index])

            X_test = torch.tensor(np.array(combine_data)[test_index], dtype=torch.float32)
            y_test = torch.tensor(np.array(combine_label)[test_index])

            # train
            epochs = 100

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                train_data = X_train.to(device)
                train_label = y_train.to(device)

                train_outputs = model(train_data) # LSTM
                # print(outputs)
                
                # Compute loss
                loss = criterion(train_outputs, train_label)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()

                weights = model.projection.weight.data
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item()}')

            # val
            model.eval()
            
            with torch.no_grad():
                val_data = X_test.to(device)
                val_label = y_test.to(device)
                val_outputs = model(val_data)
            
            _, predicted = torch.max(val_outputs, 1)
            predicted = torch.tensor(predicted).to(device)

            correct = predicted.eq(val_label.data).sum()
            total = len(val_label)
            accuracy = correct / total
            print(f'Accuracy: {accuracy * 100:.2f}%')

            f1 = f1_score(y_test, predicted.cpu(), average='weighted')
            print(f'F1-Score: {f1 * 100:.2f}%')

            avg_acc.append(accuracy.cpu())
            avg_f1.append(f1)

        print("mean acc:", np.array(avg_acc).mean())
        print("mean f1:", np.array(avg_f1).mean())

