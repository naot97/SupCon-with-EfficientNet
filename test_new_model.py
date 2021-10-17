import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from model.efficientnet import SupConEfficient
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from utils import AverageMeter, get_learning_rate, accuracy, save_checkpoint
import time 
from sklearn.metrics import classification_report

arch_name = 'efficientnet-b2'
test_transform = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# newmodel_checkpoint_path = 'runs/old/model_best.pth.tar'
# labels = ['aov', 'apex', 'cod', 'dota', 'fortnite', 'freefire', 'lol', 'mlbb', 'overwatch', 'pubg', 'pubg_pc', 'unknown', 'wild']
# test_dataset = ImageFolder('../test_old/', transform = test_transform)
# name_csv = 'confusion_old.csv'

labels = ['aov', 'apex', 'chatting', 'cod', 'dota', 'fortnite', 'freefire', 'lol', 'mlbb', 'overwatch', 'pubg', 'pubg_pc', 'unknown', 'wild']
newmodel_checkpoint_path = 'runs/gamedetect_chatting_v2/model_best.pth.tar'
test_dataset = ImageFolder('../test/', transform = test_transform)
name_csv = 'confusion_new.csv'


test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

confusion_maxtrix = np.zeros((len(labels),len(labels)))

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print_freq = 10
verbose = 1

model = SupConEfficient(arch_name, num_classes=len(labels))
model.load_state_dict(torch.load(newmodel_checkpoint_path)['state_dict'])
model = model.to(device)

pred_total = []
target_total = []

criterion = nn.CrossEntropyLoss().to(device)

def count_confusion(output, target, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_arr = pred.cpu().detach().numpy()[0]
    target_arr = target.cpu().detach().numpy()
    for i, targeti in enumerate(target_arr):
        predi = pred_arr[i]
        confusion_maxtrix[targeti][predi] += 1
    global pred_total, target_total
    pred_total = pred_total + list(pred_arr)
    target_total = target_total + list(target_arr)

def validate(val_loader, model, criterion, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        print(i)
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss


        count_confusion(output.data, target, topk=(1,))
        err1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and verbose == True and epoch != None:
            print('Test (on test set): [{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})'.format(
                 i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))
    print('*Top 1-err {top1.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
         top1=top1, loss=losses))
    return top1.avg, losses.avg

_, val_loss = validate(test_loader, model, criterion)

print(labels)
print(confusion_maxtrix)
arr2 = np.array(labels).reshape((len(labels),1))
arr = np.hstack((arr2,confusion_maxtrix))
df = pd.DataFrame(arr, columns=['label'] + labels)
# df.to_csv(name_csv)
# for i in range(len(target_total)):
#     if target_total[i] == 2:
#         target_total[i] = 12

#     if pred_total[i] == 2:
#         pred_total[i] = 12
    
print(classification_report(target_total,pred_total, target_names=labels.remove('chatting')))