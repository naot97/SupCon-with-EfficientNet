import os

lst = ['aov', 'apex', 'chatting', 'cod', 'dota', 'fortnite', 'freefire', 'lol', 'mlbb', 'overwatch', 'pubg', 'pubg_pc', 'unknown', 'wild']

for item in lst:
    os.makedirs('test_video/' + item + '/', exist_ok=True)