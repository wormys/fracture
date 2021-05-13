"""
Date: 2021/05/10
Author: worith

"""
import os
import numpy as np
import pandas as pd


def cut_str(config_list_str):
    return config_list_str[1:-1].replace("\n", " ").strip().split(', ')


def extract_feature(csv_path, save_path):
    data = pd.read_excel(csv_path)
    data.dropna(axis=0, how='any', inplace=True)
    hidden_feature_data = data.iloc[:, 14:]
    hidden_feature = hidden_feature_data.columns.tolist()
    width_list = [i for idx, i in enumerate(hidden_feature) if idx % 2 == 1]
    height_list = list(set(hidden_feature) - set(width_list))

    data_len = hidden_feature_data[height_list]
    data_width = hidden_feature_data[width_list]

    data['Average of Fracture Length'] = data_len.mean(axis=1)
    data['Average of Fracture Width'] = data_width.mean(axis=1)

    data['Variation of Fracture Length'] = data_len.var(axis=1)
    data['Variation of Fracture Width'] = data_width.var(axis=1)

    data_len = np.array(data_len)
    data_width = np.array(data_width)
    average_areas = []
    for i in range(data_len.shape[0]):
        average_area = 0
        for j in range(data_len.shape[1]):
            average_area += data_len[i][j] * data_width[i][j]
        average_areas.append(average_area / len(hidden_feature))
    data['Average Area of Fracture'] = average_areas

    in_feature = data.iloc[:, 0:13].columns.tolist()
    out_feature = ['NPV']

    hidden_feature = data.iloc[:, 14:].columns.tolist()
    data.to_csv(save_path, index=None)

    return in_feature, out_feature, hidden_feature


def main():
    csv_path = r'../data/6_stages.xlsx'
    extract_feature(csv_path, '../data/6_stages.csv')


if __name__ == '__main__':
    main()