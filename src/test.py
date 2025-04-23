from src.data_collection import add_new_action, collect_data

if __name__ == '__main__':
    """test action"""
    new_action = input('Enter new action: ')
    if add_new_action(new_action):
        print(new_action)
        collect_data([new_action])
