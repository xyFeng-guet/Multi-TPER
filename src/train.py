import os
import torch
import time
import numpy as np
from tqdm import tqdm
from opts import get_args
from core.dataset import Dataset
from sklearn.model_selection import StratifiedKFold
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        inputs = {
            'V': data['vision'].to(device),
            'A': data['audio'].to(device),
            'T': data['text'].to(device),
            'mask': {
                'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        copy_label = label.clone().detach()
        batchsize = inputs['V'].shape[0]

        output, nce_loss = model(inputs, copy_label)

        loss_re = loss_fn(output, label)
        loss = loss_re + 0.1 * nce_loss
        losses.update(loss.item(), batchsize)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    return train_results


def test(model, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            # ids = data['id']
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output, _ = model(inputs, None)
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # if epoch == 11:
        #     calculate_u_test(pred, true)
        test_results = metrics(pred, true)

    return test_results


def main(parse_args):
    opt = parse_args

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    setup_seed(opt.seed)
    dataset = Dataset(opt.datasetName, opt.labelType).d_l

    Avg_Accuracy = []
    Avg_F1_score = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(dataset['data'], dataset['label']):
        X_train = torch.tensor(np.array(dataset['data'])[train_index], dtype=torch.float32)
        Y_train = torch.tensor(np.array(dataset['label'])[train_index])

        X_test = torch.tensor(np.array(dataset['data'])[test_index], dtype=torch.float32)
        Y_test = torch.tensor(np.array(dataset['label'])[test_index])

        model = build_model(opt).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = MetricsTop().getMetics(opt.datasetName)
        scheduler_warmup = get_scheduler(optimizer, opt.epochs)

        for epoch in range(1, opt.epochs+1):
            train_results = train(model, [X_train, Y_train], optimizer, loss_fn, epoch, metrics)
            test_results = test(model, [X_test, Y_test], optimizer, loss_fn, epoch, metrics)
            save_print_results(opt, logger, train_results, test_results)
            scheduler_warmup.step()

        Avg_Accuracy.append()
        Avg_F1_score.append()

    print("mean acc:", np.array(Avg_Accuracy).mean())
    print("mean f1:", np.array(Avg_F1_score).mean())


if __name__ == '__main__':
    main(opt)
