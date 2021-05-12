"""
Date: 2021/05/10
Author: worith

"""
import os


def cut_str(config_list_str):
    return config_list_str[1:-1].replace("\n", " ").strip().split(', ')