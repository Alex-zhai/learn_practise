# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/8/3 10:49

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import calendar
from datetime import datetime

# step1: prepare transformed csv train data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df["year"] = train_df['datetime'].apply(lambda x: x.split()[0].split("-")[0]).astype(int)
train_df["month"] = train_df['datetime'].apply(lambda x: x.split()[0].split("-")[1]).astype(int)
train_df["day"] = train_df['datetime'].apply(lambda x: x.split()[0].split("-")[2]).astype(int)
train_df = train_df.drop('datetime', axis=1)

print(train_df.head(3))
