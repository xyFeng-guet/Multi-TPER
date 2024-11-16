import os
import logging
import random
import numpy as np
import torch
from tabulate import tabulate
from scipy import stats


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def ConfigLogging(file_path):
    # 创建一个 logger
    logger = logging.getLogger("save_option_results")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename=file_path, encoding='utf8')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, result, modality, model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(
        save_path,
        'MOSEI_{}_MAE-{}_Corr-{}.pth'.format(
            modality,
            result["MAE"],
            result["Corr"]
        )
    )
    torch.save(model.state_dict(), save_file_path)


def save_print_results(opt, logger, train_re, test_re):
    if opt.datasetName in ['emotake']:
        results = [
            ["Train", train_re["Accuracy"], train_re["F1-Score"]],
            ["Test", test_re["Accuracy"], test_re["F1-Score"]]
        ]
        headers = ["Phase", "Accuracy", "F1-Score"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        logger.info(table)
    else:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_2"], train_re["Mult_acc_3"], train_re["Mult_acc_5"], train_re["F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_2"], test_re["Mult_acc_3"], test_re["Mult_acc_5"], test_re["F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        if logger is not None:
            logger.info(table)
        else:
            print(table)


def calculate_u_test(pred, label):
    pred, label = pred.squeeze().numpy(), label.squeeze().numpy()
    label_mean = np.mean(label)
    alpha = 0.05

    pred_mean = np.mean(pred)
    pred_std = np.std(pred, ddof=1)
    label_std = np.std(label, ddof=1)
    # standard_error = pred_std / np.sqrt(len(pred))
    standard_error = np.sqrt(label_std / len(label) + pred_std / len(pred))

    Z = (label_mean - pred_mean) / standard_error
    critical_value = stats.norm.ppf(1 - alpha)
    if Z >= critical_value:
        print("拒绝原假设，接受备择假设")
    else:
        print("无法拒绝原假设")
