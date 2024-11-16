import os
import torch
import time
import numpy as np
from tqdm import tqdm
from opts import get_args
from core.dataset import create_Dataset, create_DataLoader
from sklearn.model_selection import StratifiedKFold
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(opt, model, dataset, optimizer, loss_fn, epoch, metrics):
    dataLoader = create_DataLoader(opt, dataset)
    train_pbar = tqdm(dataLoader)

    losses = AverageMeter()
    y_pred, y_true = [], []
    model.train()

    for data in train_pbar:
        inputs = {
            'au': data['au'].to(device),
            'em': data['em'].to(device),
            'hp': data['hp'].to(device),
            'bp': data['bp'].to(device)
        }
        label = data['label'].to(device)
        batchsize = inputs['au'].shape[0]

        output = model(inputs)

        loss_cla = loss_fn(output, label)
        loss = loss_cla
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


def test(opt, model, dataset, optimizer, loss_fn, epoch, metrics):
    dataLoader = create_DataLoader(opt, dataset)
    test_pbar = tqdm(dataLoader)

    losses = AverageMeter()
    y_pred, y_true = [], []
    model.eval()

    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'au': data['au'].to(device),
                'em': data['em'].to(device),
                'hp': data['hp'].to(device),
                'bp': data['bp'].to(device)
            }
            label = data['label'].to(device)
            batchsize = inputs['au'].shape[0]

            output = model(inputs)

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

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
    dataset = create_Dataset(opt.datasetName, opt.labelType).d_l

    Avg_Accuracy = []
    Avg_F1_score = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(dataset['data'], dataset['label']):
        X_train = np.array(dataset['data'])[train_index]
        Y_train = np.array(dataset['label'])[train_index]

        X_test = np.array(dataset['data'])[test_index]
        Y_test = np.array(dataset['label'])[test_index]

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
            train_results = train(opt, model, [X_train, Y_train], optimizer, loss_fn, epoch, metrics)
            test_results = test(opt, model, [X_test, Y_test], optimizer, loss_fn, epoch, metrics)
            save_print_results(opt, logger, train_results, test_results)
            scheduler_warmup.step()

        Avg_Accuracy.append(test_results['Accuracy'])
        Avg_F1_score.append(test_results['F1-Score'])

    logger.info(f"mean acc: {np.array(Avg_Accuracy).mean()}")
    logger.info(f"mean f1 : {np.array(Avg_F1_score).mean()}")


if __name__ == '__main__':
    main(opt)
