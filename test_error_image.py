import os
import cv2
from skimage import io


datadir = '../test/'
game_lst = ['aov','apex','cod', 'dota', 'fortnite', 'freefire', 'lol', 'mlbb', 'overwatch', 'pubg', 'pubg_pc', 'unknown', 'wild']
#game_lst = ['unknown']
def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

for game in game_lst:
    game_dir = datadir + game + '/'
    file_lst = [f for f in os.listdir(game_dir)]
    print(game,len(file_lst))
    # for file in file_lst:
    #     if verify_image(game_dir + file) == False:
    #         print(file)
    #         os.remove(game_dir + file)