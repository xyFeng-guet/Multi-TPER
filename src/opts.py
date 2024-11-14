import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Multi-TPER')

    # Common
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='optimizer learning rate')

    # Dataset
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='data file')

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
