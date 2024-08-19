import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import random

class HandDataset(Dataset):
    def __init__(self, dataframe, labels, image_dir, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
            raise IndexError("Index out of bounds")

        anchor_img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['imageName'])
        anchor_image = Image.open(anchor_img_name).convert('RGB')
        anchor_label = self.labels.iloc[idx].astype(float)

        # Find a positive example
        same_id_df = self.dataframe[(self.dataframe['id'] == self.dataframe.iloc[idx]['id']) &
                                    (self.dataframe['p'] == self.dataframe.iloc[idx]['p']) &
                                    (self.dataframe['r'] == self.dataframe.iloc[idx]['r'])]
        if same_id_df.empty:
            raise ValueError("No positive example found for the anchor")
        positive_idx = random.choice(same_id_df.index)
        positive_img_name = os.path.join(self.image_dir, self.dataframe.loc[positive_idx]['imageName'])
        positive_image = Image.open(positive_img_name).convert('RGB')

        # Find a negative example
        different_id_df = self.dataframe[(self.dataframe['id'] != self.dataframe.iloc[idx]['id']) & 
                                         (self.dataframe['p'] == self.dataframe.iloc[idx]['p']) &
                                         (self.dataframe['r'] == self.dataframe.iloc[idx]['r'])]
        if different_id_df.empty:
            raise ValueError("No negative example found for the anchor")
        negative_idx = random.choice(different_id_df.index)
        negative_img_name = os.path.join(self.image_dir, self.dataframe.loc[negative_idx]['imageName'])
        negative_image = Image.open(negative_img_name).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, torch.tensor(anchor_label)


class Make_dataset:
    def __init__(self, image_directory='Hands', info='HandInfo.csv', batch_size=5, 
                 img_height=224, img_width=224, split_size=0.5):
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.split_size = split_size
        self.info = info
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
        self.df = df
        df = df[df.accessories == 0]
        df['p'] = np.where(df.aspectOfHand.str.startswith('p'), 1, 0)
        df['r'] = np.where(df.aspectOfHand.str.endswith('right'), 1, 0)
        self.df_p_r = df[(df.p == 1) & (df.r == 1)]
        self.df_p_l = df[(df.p == 1) & (df.r == 0)]
        self.df_d_r = df[(df.p == 0) & (df.r == 1)]
        self.df_d_l = df[(df.p == 0) & (df.r == 0)]        

    def split_data(self):
        combined_df = pd.concat([self.df_p_r, self.df_p_l, self.df_d_r, self.df_d_l])
        subjects = combined_df['id'].unique()
        train_subjects, val_subjects = train_test_split(subjects, test_size=self.split_size, random_state=42)
        self.train_df = combined_df[combined_df['id'].isin(train_subjects)]
        self.val_df = combined_df[combined_df['id'].isin(val_subjects)]
        self.train_df.reset_index()
        self.val_df.reset_index()
        return self.train_df, self.val_df

    def encoder(self):
        self.define_data_frames()
        self.split_data()
        enc = LabelEncoder()
        enc.fit(self.df.id)
        self.train_df['id'] = enc.transform(self.train_df['id'])
        self.val_df['id'] = enc.transform(self.val_df['id'])
        self.num_classes1 = len(enc.classes_)

def create_datasets():
    make_dataset = Make_dataset()
    make_dataset.define_data_frames()
    make_dataset.encoder()
    # Create datasets
    make_dataset.val_p_r = make_dataset.val_df[(make_dataset.val_df.p == 1) & (make_dataset.val_df.r == 1)]
    make_dataset.val_p_l = make_dataset.val_df[(make_dataset.val_df.p == 1) & (make_dataset.val_df.r == 0)]
    make_dataset.val_d_r = make_dataset.val_df[make_dataset.val_df.p == 0 & (make_dataset.val_df.r == 1)]
    make_dataset.val_d_l = make_dataset.val_df[(make_dataset.val_df.p == 0) & (make_dataset.val_df.r == 0)]

    gallery_p_r = make_dataset.val_p_r.groupby('id').apply(lambda x: x.sample(1)).reset_index(drop=True)
    gallery_p_l = make_dataset.val_p_l.groupby('id').apply(lambda x: x.sample(1)).reset_index(drop=True)
    gallery_d_r = make_dataset.val_d_r.groupby('id').apply(lambda x: x.sample(1)).reset_index(drop=True)
    gallery_d_l = make_dataset.val_d_l.groupby('id').apply(lambda x: x.sample(1)).reset_index(drop=True)

    train_dataset = HandDataset(make_dataset.train_df, 
                                make_dataset.train_df.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['train'])
    
    val_dataset = HandDataset(make_dataset.val_df, 
                                make_dataset.val_df.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    val_dataset_p_r = HandDataset(make_dataset.val_p_r, 
                                make_dataset.val_p_r.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    val_dataset_p_l = HandDataset(make_dataset.val_p_l, 
                                make_dataset.val_p_l.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    val_dataset_d_r = HandDataset(make_dataset.val_d_r, 
                                make_dataset.val_d_r.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    val_dataset_d_l = HandDataset(make_dataset.val_d_l, 
                                make_dataset.val_d_l.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    
    gal_dataset_p_r = HandDataset(gallery_p_r, 
                                gallery_p_r.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    gal_dataset_p_l = HandDataset(gallery_p_l, 
                                gallery_p_l.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    gal_dataset_d_r = HandDataset(gallery_d_r, 
                                gallery_d_r.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    gal_dataset_d_l = HandDataset(gallery_d_l, 
                                gallery_d_l.id, 
                                make_dataset.image_directory, transform=make_dataset.data_transforms['val'])
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=make_dataset.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=make_dataset.batch_size, shuffle=False)
    val_loader_p_r = DataLoader(val_dataset_p_r, batch_size=make_dataset.batch_size, shuffle=False)
    val_loader_p_l = DataLoader(val_dataset_p_l, batch_size=make_dataset.batch_size, shuffle=False)
    val_loader_d_r = DataLoader(val_dataset_d_r, batch_size=make_dataset.batch_size, shuffle=False)
    val_loader_d_l = DataLoader(val_dataset_d_l, batch_size=make_dataset.batch_size, shuffle=False)

    gal_loader_p_r = DataLoader(gal_dataset_p_r, batch_size=make_dataset.batch_size, shuffle=False)
    gal_loader_p_l = DataLoader(gal_dataset_p_l, batch_size=make_dataset.batch_size, shuffle=False)
    gal_loader_d_r = DataLoader(gal_dataset_d_r, batch_size=make_dataset.batch_size, shuffle=False)
    gal_loader_d_l = DataLoader(gal_dataset_d_l, batch_size=make_dataset.batch_size, shuffle=False)

    vals = [val_loader_p_r, val_loader_p_l, val_loader_d_r, val_loader_d_l]
    gals = [gal_loader_p_r, gal_loader_p_l, gal_loader_d_r, gal_loader_d_l]
    return train_loader, val_loader, make_dataset, vals, gals


