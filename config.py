import argparse

args = argparse.ArgumentParser()

# general
args.add_argument('--batch_size', type=int, default=1024)
args.add_argument('--epochs', type=int, default=64)
args.add_argument('--seed', type=int, default=2024)

args.add_argument('--num_samples', type=int, default=None)  # None, int
args.add_argument('--is_train', type=int, default=1)  # True, False
args.add_argument('--is_valid', type=int, default=1)

args.add_argument('--dataset', type=str,
                  default='kuairand_pure',
                  choices=['kuairand_pure', 'kuairand_1k'])

args.add_argument('--cold_start', type=int, default=0, choices=[0, 1])

args.add_argument('--model_name', type=str,
                  default='OURS', choices=['OURS', 'TaFR', 'TCCM'])

args.add_argument('--backbone', type=str,
                  default='AFM', choices=['NFM', 'MLP', 'WideDeep', 'DeepFM', 'AFM'])

args.add_argument('--strategy', type=str,
                  default='raw', choices=['raw', 'smooth'])

args.add_argument('--pred_mode', type=str,
                  default='!', choices=['do', '!'])

args.add_argument('--beta', type=float, default=0.5)

args.add_argument('--gamma', type=float, default=0.5)

args.add_argument('--tafr_acc', type=str, default='-1.5')

args.add_argument('--load_epoch', type=int, default=64)
args = args.parse_args()



