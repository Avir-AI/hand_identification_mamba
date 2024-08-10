import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

class HandDataset(Dataset):
    def __init__(self, dataframe, labels, image_dir, transform=None):
        self.dataframe = dataframe
        self.labels1, self.labels2, self.labels3 = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['imageName'])
        image = Image.open(img_name).convert('RGB')
        label1 = self.labels1.iloc[idx].values.astype(float)
        label2 = self.labels2.iloc[idx].values.astype(float)
        label3 = self.labels3.iloc[idx].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, [torch.tensor(label1), torch.tensor(label2), torch.tensor(label3)]
    
class HandDataset_Gal(Dataset):
    def __init__(self, dataframe, labels, image_dir, transform=None):
        self.dataframe = dataframe
        self.labels1, self.labels2, self.labels3 = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['imageName'])
        image = Image.open(img_name).convert('RGB')
        label = self.labels1.iloc[idx].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, [torch.tensor(label)]


class Make_dataset:
    def __init__(self, image_directory = 'Hands', info = 'HandInfo.csv', batch_size = 20, 
                 img_height = 224, img_width = 224, split_size = 0.2):
        #parameters
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.split_size = split_size
        self.info = info
        # Define the data augmentation and normalization transforms
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
    
    def define_data_frames(self):
        df = pd.read_csv(self.info)
        df['idx'] = df.index
        age_bins = [0, 21, 22, 23, 24, 31, 76]
        labels = np.arange(6)
        df['age_category'] = pd.cut(df['age'], bins=age_bins, labels=labels, right=False, include_lowest=True)
        df = df[df.accessories == 0]
        df['p'] = np.where(df.aspectOfHand.str.startswith('p') == True, 1, 0)
        df['r'] = np.where(df.aspectOfHand.str.endswith('right') == True, 1, 0)
        self.df_p_r = df[(df.p == 1) & (df.r == 1)]
        self.df_p_l = df[(df.p == 1) & (df.r == 0)]
        self.df_d_r = df[(df.p == 0) & (df.r == 1)]
        self.df_d_l = df[(df.p == 0) & (df.r == 0)]        

    # Split data into training and validation sets
    def split_data(self, df):
        self.train_df, self.val_df = df.iloc[:int(df.shape[0] / 2), :], df.iloc[int(df.shape[0] / 2):, :]
        return self.train_df, self.val_df

    def encode(self, col, encoder, val_df, gal_df, train_df, sambol):
        encoder.fit(sambol[col])
        train_df[col] = encoder.transform(train_df[col])
        val_df[col] = encoder.transform(val_df[col])
        gal_df[col] = encoder.transform(gal_df[col])
    
    def encoder(self):
        train_df_p_r, val_df_p_r = self.split_data(self.df_p_r)
        train_df_p_l, val_df_p_l = self.split_data(self.df_p_l)
        train_df_d_r, val_df_d_r = self.split_data(self.df_d_r)
        train_df_d_l, val_df_d_l = self.split_data(self.df_d_l)
        np.random.seed(42)
        gall_df_p_r = val_df_p_r.groupby('id', group_keys=False).apply(lambda x: x.sample(1))
        gall_df_p_l = val_df_p_l.groupby('id', group_keys=False).apply(lambda x: x.sample(1))
        gall_df_d_r = val_df_d_r.groupby('id', group_keys=False).apply(lambda x: x.sample(1))
        gall_df_d_l = val_df_d_l.groupby('id', group_keys=False).apply(lambda x: x.sample(1))
        sambol = pd.concat([self.df_p_r, self.df_p_l, self.df_d_r, self.df_d_l])
        train_df = pd.concat([train_df_p_r, train_df_d_r, train_df_p_l, train_df_d_l])
        val_df = pd.concat([val_df_p_r, val_df_d_r, val_df_p_l, val_df_d_l])
        gal_df = pd.concat([gall_df_p_r, gall_df_p_l, gall_df_d_r, gall_df_d_l])
        self.val_dic = {'Total': val_df, 'Palmer Right': val_df_p_r, 'Palmer Left': val_df_p_l, 'Dorsal Right': val_df_d_r, 'Dorsal Left': val_df_d_l}
        self.gal_dic = {'Total': gal_df, 'Palmer Right': gall_df_p_r, 'Palmer Left': gall_df_p_l, 'Dorsal Right': gall_df_d_r, 'Dorsal Left': gall_df_d_l}
        pairs = [('id', LabelEncoder()), ('age_category', LabelEncoder()), ('gender', LabelEncoder())]
        for pair in pairs:
            self.encode(pair[0], pair[1], val_df, gal_df, train_df, sambol)
        self.train_id_one_hot = pd.get_dummies(train_df['id'])
        self.val_id_one_hot = pd.get_dummies(val_df['id'])
        self.gal_id_one_hot = pd.get_dummies(gal_df['id'])
        self.train_age_one_hot = pd.get_dummies(train_df['age_category'])
        self.val_age_one_hot = pd.get_dummies(val_df['age_category'])
        self.train_gender_one_hot = pd.get_dummies(train_df['gender'])
        self.val_gender_one_hot = pd.get_dummies(val_df['gender'])
        self.num_classes1 = len(pairs[0][1].classes_)
        self.num_classes2 = len(pairs[1][1].classes_)
        self.num_classes3 = len(pairs[2][1].classes_)

def create_datasets(infer = None):
    make_dataset = Make_dataset()
    if not infer:
        make_dataset.define_data_frames()
        make_dataset.encoder()
        # Create datasets
        train_dataset = HandDataset(make_dataset.train_df, 
                                    [make_dataset.train_id_one_hot, make_dataset.train_age_one_hot, make_dataset.train_gender_one_hot], 
                                    make_dataset.image_directory, transform=make_dataset.data_transforms['train'])
        val_dataset = HandDataset(make_dataset.val_df, 
                                [make_dataset.val_id_one_hot, make_dataset.val_age_one_hot, make_dataset.val_gender_one_hot], 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=make_dataset.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=make_dataset.batch_size, shuffle=False)
        return train_loader, val_loader, make_dataset
    else:
        dataset = datasets.ImageFolder(root=infer, transform=make_dataset.data_transforms['val'])
        val_loader = DataLoader(dataset, batch_size=make_dataset.batch_size, shuffle=False)
        return None, val_loader, make_dataset
