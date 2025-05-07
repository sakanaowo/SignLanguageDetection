import re

from src.data_collection import add_new_action, collect_data
from utils.helper import extract_log, plot_loss

if __name__ == '__main__':
    """test action"""
    # new_action = input('Enter new action: ')
    # if add_new_action(new_action):
    #     print(new_action)
    #     collect_data([new_action])


    with open('../outputs/logs/3-11-5-7-log.txt', 'r', encoding='utf-8') as f:
        log = f.read()

    log_data = extract_log(log)
    print(log_data)
    # plot_loss(log_data)
