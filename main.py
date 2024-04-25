import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from preprocessing.loader import Loader
from preprocessing.generator import Generator
from utils.train import *
from utils.seed import set_seed
from config import args
from models.OURS import OURS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if __name__ == '__main__':
    # set seed
    set_seed(args.seed)

    print('Device:', device)

    data = Generator(args.dataset, args.model_name, tafr_acc=args.tafr_acc, cold_start=args.cold_start)
    train_loader, test_loader, train_df, test_df, confounder_info \
        = data.wrapper(batch_size=args.batch_size, num_samples=args.num_samples)
    feature_size_map = getattr(data, 'feature_size_map')
    feature_size_map_item = getattr(data, 'feature_size_map_item')
    n_day = 30

    model = OURS(device=device,
                 model_name=args.model_name,
                 strategy=args.strategy,
                 feature_size_map=feature_size_map,
                 feature_size_map_item=feature_size_map_item,
                 n_day=n_day,
                 embed_dim_sparse=16,
                 backbone=args.backbone)

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train and test model
    TrainAndTest(model=model, model_name=args.model_name, backbone=args.backbone, device=device, strategy=args.strategy,
                 optimizer=optimizer, criterion=criterion, seed=args.seed, cold=args.cold_start,
                 train_loader=train_loader, test_loader=test_loader, train_df=train_df, test_df=test_df,
                 dataset=args.dataset, num_samples=args.num_samples, epochs=args.epochs,
                 is_train=args.is_train, is_valid=args.is_valid, load_epoch=args.load_epoch,
                 confounder_info=confounder_info, pred_mode=args.pred_mode, params_dict=None,
                 alpha=0.3, beta=args.beta, gamma=args.gamma
                )
