import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Multi-TPER')

    # Common
    parser.add_argument('--log_path', type=str, default='./log/',
                        help='log path of train and test')
    parser.add_argument('--datasetName', type=str, default='emotake',
                        help='dataset name of use training')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='optimizer learning rate')

    # Dataset
    parser.add_argument('--labelType', type=str, default='readiness',
                        help='label type is one of quality / readiness / ra')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='data file')
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='data loader num workers')

    # Modal
    parser.add_argument('--seq_len', type=int, default=300,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='subset for M4')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='inverse output data')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='mask ratio')
    parser.add_argument('--anomaly_ratio', type=float, default=0.25,
                        help='prior anomaly ratio (%)')
    parser.add_argument('--pred_type', type=str, default='quality',
                        help='options: [quality, ra, readiness]')
    parser.add_argument('--num_class', type=int, default=3,
                        help='quality:3, ra:3, readiness:2')

    args = parser.parse_args()
    return args
