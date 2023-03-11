import os
import sys
sys.path.append('/kaggle/input/timm-0-6-9/pytorch-image-models-master')
import numpy as np
import pandas as pd
import random
import gc
import cv2
from tqdm import tqdm
import time
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import logging
from PIL import Image
import pickle
from scipy.optimize import minimize
import pickle
from sklearn.preprocessing import LabelEncoder


CFG = {
    'seed': 42,
    'model': 'tf_efficientnet_b0.ap_in1k', # resnet50, tf_efficientnet_b5_ap
    'img_size': 256,
    'resize': 256, 
    'epochs': 25,
    'train_bs': 256, 
    'valid_bs': 256,
    'epochs_warmup': 1, 
    'lr': 1e-3, 
    'weight_decay': 1e-6,
    'num_workers': 90,
    'criterion':nn.BCEWithLogitsLoss()
}

is_kaggle_env =False


if is_kaggle_env:
    MAIN_DIR="/kaggle/input/nfl-player-contact-detection"
else:
    MAIN_DIR="/home/ecs-user/data/nfl-player-contact-detection"
        
logging.basicConfig(filename='logger.log', level=logging.INFO)
logging.info('******************* CFG *******************')
logging.info(CFG)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_aug = A.Compose([
    A.Resize(height=CFG['resize'],width=CFG['resize'], p=1),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.25),
    A.Cutout(num_holes=8, max_h_size=12, max_w_size=12, fill_value=0, p=0.5),
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
    ])

valid_aug = A.Compose([
    A.Resize(height=CFG['resize'], width=CFG['resize'], p=1),
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])

with open('/home/ecs-user/nfl-player-contact-detection/notebook/exp3/video2helmets.pkl', 'rb') as f:
    video2helmets = pickle.load(f)
with open('/home/ecs-user/nfl-player-contact-detection/notebook/exp3/video2frames.pkl', 'rb') as f:
    video2frames = pickle.load(f)
    
# =============================================================================
# def
# =============================================================================
def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols

def create_previous_step_diff(df):
    grouped = df.groupby("nfl_player_id")
    df['previous_speed'] = grouped["speed"].diff().fillna(0)
    df['previous_acceleration'] = grouped["acceleration"].diff().fillna(0)
    df['previous_orientation'] = grouped["orientation"].diff().fillna(0)
    df['previous_direction'] = grouped["direction"].diff().fillna(0)
    df['previous_x_position'] = grouped["x_position"].diff().fillna(0)
    df['previous_y_position'] = grouped["y_position"].diff().fillna(0)
    df['previous_sa'] = grouped["sa"].diff().fillna(0)
    
    df['next_speed'] = grouped["speed"].diff(-1).fillna(0)
    df['next_acceleration'] = grouped["acceleration"].diff(-1).fillna(0)
    df['next_orientation'] = grouped["orientation"].diff(-1).fillna(0)
    df['next_direction'] = grouped["direction"].diff(-1).fillna(0)
    df['next_x_position'] = grouped["x_position"].diff(-1).fillna(0)
    df['next_y_position'] = grouped["y_position"].diff(-1).fillna(0)
    df['next_sa'] = grouped["sa"].diff().fillna(0)
    
    return df
    
def create_diff_features(df):
    df["diff_speed"] = df["speed_1"] - df["speed_2"]
    df["diff_distance"] = df["distance_1"] - df["distance_2"]
    df["diff_direction"] = df["direction_1"] - df["direction_2"]
    df["diff_orientation"] = df["orientation_1"] - df["orientation_2"]
    # df["diff_x_position"] = df["x_position_1"] - df["x_position_2"]
    # df["diff_y_position"] = df["y_position_1"] - df["y_position_2"]
    # df["diff_acceleration"] = df["acceleration_1"] - df["acceleration_2"]
    return df

def distance_filtere(df): 
    df_filtered = df.query('not distance>2').reset_index(drop=True)
    return df_filtered

def evaluate(model, loader_val, *, compute_score=True, pbar=None):
    """
    Predict and compute loss and score
    """
    tb = time.time()
    was_training = model.training
    model.eval()
    criterion = CFG['criterion']

    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []

    if pbar is not None:
        pbar = tqdm(desc='Predict', nrows=78, total=pbar)

    # for batch in loader_val:
    for ibatch, batch in enumerate(tqdm(loader_val)):
        img, feature, y = [x.to(device) for x in batch]
        n = y.size(0)
        with torch.no_grad():
            y_pred = model(img, feature).squeeze(-1)
        loss = criterion(y_pred.view(-1), y)

        n_sum += n
        loss_sum += n * loss.item()

        y_all.append(y.cpu().detach().numpy())
        y_pred_all.append(y_pred.cpu().detach().numpy())

        if pbar is not None:
            pbar.update(len(img))
        
        del loss, y_pred, img, y
        gc.collect()

    loss_val = loss_sum / n_sum

    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)
    
    # ==============================
    # optimize
    # ==============================
    def func(x0, y, y_pred):
        score = matthews_corrcoef(y, y_pred>x0)
        return -score

    x0 = [0.5]
    result = minimize(func, x0, args=(y, y_pred),  method="nelder-mead")
    threshold = result.x[0]
    print("score:", round(matthews_corrcoef(y, y_pred>threshold), 5))
    print("threshold", round(threshold, 5))

    score = matthews_corrcoef(y, (y_pred>threshold).astype(int)) if compute_score else None

    ret = {'loss': loss_val,
           'score': score,
           'y': y,
           'y_pred': y_pred,
           'time': time.time() - tb,
           'threshold': round(threshold, 5),
           }
    
    model.train(was_training)  # back to train from eval if necessary
    gc.collect()
    return ret

class MyDataset(Dataset):
    def __init__(self, df, feature_cols, video2helmets, video2frames, aug=valid_aug, scaler=None, mode='train'):    
        self.df = df
        self.frame = df.frame.values
        self.feature = scaler.transform(df[feature_cols].fillna(-1))
        self.players = df[['nfl_player_id_1','nfl_player_id_2']].values
        self.game_play = df.game_play.values
        self.aug = aug
        self.mode = mode
        
        self.video2helmets = video2helmets
        self.video2frames = video2frames
        
    def __len__(self):
        return len(self.df)
    
    # @lru_cache(1024)
    # def read_img(self, path):
    #     return cv2.imread(path, 0)
   
    def __getitem__(self, idx):   
        step = 5
        window = 7 * step
        frame = self.frame[idx]
        
        # if self.mode == 'train':
        #     frame = frame + random.randint(-6, 6)

        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))
            # break
        
        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'

            tmp = self.video2helmets[video]
            tmp = tmp[tmp['frame'].between(frame-window, frame+window)]
            tmp = tmp[tmp.nfl_player_id.isin(players)]
            
            # mask
            # box_df = tmp[['nfl_player_id', 'frame', 'left','width','top','height']].set_index(['frame', 'nfl_player_id', ])
            
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby('frame')[['left','width','top','height']].mean()

            bboxes = []
            for f in range(frame-window, frame+window+1, step):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][['left','width','top','height']]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            ## TODO: kosokuka
            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
            bboxes = bboxes[::1]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0
                    
            for i, f in enumerate(range(frame-window, frame+window+1, step)):
                img_new = np.zeros((256, 256), dtype=np.float32)
                
                if flag == 1 and f <= self.video2frames[video]:
                    # img = cv2.imread(MAIN_DIR + f'/frames/{video}_{f:04d}.jpg', 0)
                    ## TODO: kosokuka
                    with open(MAIN_DIR + f'/frames_gray_pkl/{video}_{f:04d}.pkl', mode='rb') as f:
                        img = pickle.load(f)

                    # # mask
                    # if f in tmp_frames:
                    #     mask = box_df.loc[f][['left','width','top','height']].values.astype(int)
                    #     for x, w, y, h in  mask:
                    #         cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4, shift=0)
                    
                    x, w, y, h = bboxes[i]

                    crop_size = 6
                    img = img[max(0, int(y+h/2)-int(crop_size*h)):int(y+h/2)+int(crop_size*h),max(0, int(x+w/2)-int(crop_size*w)):int(x+w/2)+int(crop_size*w)].copy()
                    img = cv2.resize(img, (256, 256))
                    img_new[:img.shape[0], :img.shape[1]] = img
                    
                imgs.append(img_new)
                
        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)    
        img = self.aug(image=img)["image"]
        label = np.float32(self.df.contact.values[idx])

        return img, feature, label

# =============================================================================
# CNN
# =============================================================================
class Model(nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.backbone = timm.create_model(CFG['model'], pretrained=True, features_only=True, in_chans=3, out_indices=(4,5))
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64+4096, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(320, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2) # torch.Size([1, 256, 2, 3, 3])
        )

    def forward(self, img, feature):
        b, c, h, w = img.shape
        img = img.reshape(b*2, c//2, h, w)
        img1 = img[:, 0:3, :, :] # torch.Size([128, 3, 256, 256])
        img2 = img[:, 3:6, :, :]
        img3 = img[:, 6:9, :, :]
        img4 = img[:, 9:12, :, :]
        img5 = img[:, 12:15, :, :]
        
        img1 = self.backbone(img1)[-1].view(b*2, 320, -1, 8, 8)
        img2 = self.backbone(img2)[-1].view(b*2, 320, -1, 8, 8)
        img3 = self.backbone(img3)[-1].view(b*2, 320, -1, 8, 8)
        img4 = self.backbone(img4)[-1].view(b*2, 320, -1, 8, 8)
        img5 = self.backbone(img5)[-1].view(b*2, 320, -1, 8, 8)
        
        img = torch.cat([img1, img2, img3, img4, img5], axis=2)
        img = self.conv3d(img).reshape(b, -1)

        feature = self.mlp(feature)
        y = self.fusion(torch.cat([img, feature], dim=1))
        return y

# =============================================================================
# main
# =============================================================================
def main():
    train = pd.read_csv(os.path.join(MAIN_DIR, 'train_labels.csv'), parse_dates=["datetime"])
    labels = expand_contact_id(pd.read_csv(MAIN_DIR + "/sample_submission.csv"))

    train_tracking = pd.read_csv(MAIN_DIR + "/train_player_tracking.csv")
    test_tracking = pd.read_csv(MAIN_DIR + "/test_player_tracking.csv")
    
    use_cols = [
        'x_position', 'y_position', 'speed', 'distance',
        'direction', 'orientation', 'acceleration', 'sa'
    ]
    
    cols = ["team", "position"]
    for col in cols:
        le = LabelEncoder()
        le.fit(train_tracking[col])
        train_tracking[col] = le.transform(train_tracking[col])

    train_tracking = create_previous_step_diff(train_tracking)
    
    use_cols+=[
        'previous_speed', 
        'previous_acceleration', 
        'previous_orientation', 
        'previous_direction', 
        'previous_x_position', 
        'previous_y_position', 
        'previous_sa',
        'next_speed', 
        'next_acceleration', 
        'next_orientation', 
        'next_direction', 
        'next_x_position', 
        'next_y_position', 
        'next_sa',
        "team", 
        "position"]
    
    train, feature_cols = create_features(train, train_tracking, use_cols=use_cols)
    train = create_diff_features(train)
    feature_cols += [
        "diff_speed", 
        "diff_direction", 
        "diff_orientation", 
        "diff_distance"]
    
    train['frame'] = (train['step']/10*59.94+5*59.94).astype('int')+1
    train['game_key'] = train.game_play.apply(lambda x:x.split('_')[0])
    train_filtered = distance_filtere(train)

    train_game_key = train_filtered.game_key.unique()[:135]
    val_game_key = train_filtered.game_key.unique()[135:]
    train_df = train_filtered[train_filtered['game_key'].isin(train_game_key)].reset_index(drop=True)
    val_df = train[train['game_key'].isin(val_game_key)].reset_index(drop=True)    
    
    ## Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(-1))
    
    val_set = MyDataset(val_df, feature_cols, video2helmets, video2frames, valid_aug, scaler, 'test')
    val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)
    
    num_features = len(feature_cols)
    model = Model(num_features)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model.to(device)
    model.train()
    
    time_val = 0.0
    lrs = []

    tb = time.time()
    print('Epoch   loss          score   lr')
    logging.info('******************* MAIN *******************')
    logging.info('Epoch   loss          score   lr')
    best_val_result = None
    early_stopping_count = 0
    best_loss = 99999
    best_score = 0
    for iepoch in range(CFG['epochs']):
        # sampling
        logging.info(f'before sampling train_df: {train_df.shape}')
        contact_df = train_df[train_df['contact']==1]
        noncontact_df = train_df[train_df['contact']==0].sample(n=len(contact_df)*3, random_state=42)
        train_df = pd.concat([contact_df, noncontact_df]).reset_index(drop=True)
        logging.info(f'after sampling train_df: {train_df.shape}')

        train_set = MyDataset(
            train_df, 
            feature_cols, 
            video2helmets, 
            video2frames, 
            train_aug, 
            scaler, 
            'train')
        train_loader = DataLoader(
            train_set, 
            batch_size=CFG['train_bs'], 
            shuffle=True, 
            num_workers=CFG['num_workers'], 
            pin_memory=True, 
            drop_last=True)
        
        # Learning-rate schedule
        nbatch = len(train_loader)
        warmup = CFG['epochs_warmup'] * nbatch  # number of warmup steps
        nsteps = CFG['epochs'] * nbatch        # number of total steps
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        criterion = CFG['criterion']
        
        max_grad_norm=1000
        scheduler = CosineLRScheduler(
            optimizer,
            warmup_t=warmup, 
            warmup_lr_init=0.0, 
            warmup_prefix=True, # 1 epoch of warmup
            t_initial=(nsteps - warmup), 
            lr_min=1e-6
            )
        
        loss_sum = 0.0
        n_sum = 0

        # Train
        grad_scaler = GradScaler()
        for ibatch, batch in enumerate(tqdm(train_loader)):
            img, feature, y = [x.to(device) for x in batch]
            n = y.size(0)
            
            optimizer.zero_grad()
            with autocast():
                y_pred = model(img, feature).squeeze(-1)
                loss = criterion(y_pred.view(-1), y)
                loss_train = loss.item()
            
            loss_sum += n * loss_train
            n_sum += n

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step(iepoch * nbatch + ibatch + 1)
            lrs.append(optimizer.param_groups[0]['lr'])          

        # Evaluate
        val = evaluate(model, val_loader)
        time_val += val['time']
        loss_train = loss_sum / n_sum
        lr_now = optimizer.param_groups[0]['lr']
        dt = (time.time() - tb) / 60
        print('Epoch %d %.4f %.4f %.4f  %.2e  %.2f min' %
            (iepoch + 1, loss_train, val['loss'], val['score'], lr_now, dt))
        logging.info('Epoch %d %.4f %.4f %.4f  %.2e  %.2f min' % (iepoch + 1, loss_train, val['loss'], val['score'], lr_now, dt))
        logging.info(f'val threshold: {val["threshold"]}')
        
        if val['score'] > best_score:
            best_score = val['score']
            # Save model
            ofilename =f'model_pretrain_epoch{iepoch}.pytorch'
            torch.save(model.module.state_dict(), ofilename)
            print(ofilename, 'written')
            early_stopping_count = 0
        else:
            early_stopping_count+=1
        del val
        gc.collect()

    dt = time.time() - tb
    print('Training done %.2f min total, %.2f min val' % (dt / 60, time_val / 60))

    gc.collect()

if __name__ == '__main__':
    main()