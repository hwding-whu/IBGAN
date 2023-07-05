from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch, gc
from torchvision.models import densenet121, densenet201, resnet18, MobileNetV3, mobilenet_v3_large, resnet152, vgg11
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import os
import csv
import seaborn as sns
import numpy as np
from collections import OrderedDict
import random
#随机种子
# random.seed(2)
#cuda设置
# os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
# gc.collect()
# torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = r'./weights/'
#参数设置
print (torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(file, n):
    save_path = 'logs/bloodmnist/Original'
    if os.path.exists(save_path):
        True
    else:
        os.makedirs(save_path)
    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    train_ds = ImageFolder("./data/" + file + "/Training", transform=data_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_ds = ImageFolder("./data/" + file + "/Test", transform=data_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    # print(train_ds[0][0].size())
    # print(train_ds.class_to_idx)
    # print(train_ds.imgs)
    label = np.arange(len(train_ds.class_to_idx)).tolist()
    EMOS = ['{}'.format(x) for x in label]
    # 训练过程
    total_step = len(train_ds)
    for t in range(n):
        print('第%d轮训练&测试' % (t))
        model = resnet18(weights=True)
        model.fc = torch.nn.Linear(512, len(train_ds.class_to_idx))
        # num_ftrs = model.classifier.in_features
        # model.classifier = torch.nn.Linear(num_ftrs, 10)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(Epochs):
            print("===========%s_%d: Epoch:%d==========" % (file, t, epoch))
            # model.train()
            loop = tqdm(train_dl)
            total_loss = 0
            train_preds = []
            train_trues = []
            for i, (images, labels) in enumerate(loop):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fc(outputs, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    step_loss = loss.data.item()
                    total_loss += step_loss
                    train_outputs = outputs.argmax(dim=1)
                    train_preds.extend(train_outputs.detach().cpu().numpy())
                    train_trues.extend(labels.detach().cpu().numpy())
            with torch.no_grad():
                # print (train_preds)
                accuracy = accuracy_score(train_trues, train_preds)
                precision = precision_score(train_trues, train_preds, average='macro')
                recall = recall_score(train_trues, train_preds, average='macro')
                f1 = f1_score(train_trues, train_preds, average='macro')
                print(
                    "[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                        epoch, total_loss / total_step, accuracy, precision, recall, f1))
        # torch.save(model, save_path + '/model_%d.pkl' % (t))

        # 测试过程
        total_num = len(test_dl.dataset)
        print(total_num, len(test_dl))
        test_preds = []
        test_trues = []
        with torch.no_grad():
            model.eval()
            for data, target in test_dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_outputs = output.argmax(dim=1)
                test_preds.extend(test_outputs.detach().cpu().numpy())
                test_trues.extend(target.detach().cpu().numpy())
            accuracy = accuracy_score(test_trues, test_preds)
            precision = precision_score(test_trues, test_preds, average='macro')
            recall = recall_score(test_trues, test_preds, average='macro')
            f1 = f1_score(test_trues, test_preds, average='macro')
            cm = confusion_matrix(test_trues, test_preds, labels=label)
            print('======================测试结果=======================')
            print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                accuracy, precision, recall, f1))

        print('======================混淆矩阵=======================')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        cm = np.around(cm, decimals=2)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(save_path + '/heatmap_%d.jpg' % (t))
        # plt.show()
        print('======================测试报告=======================')
        NUM_EMO = len(EMOS)
        cr = classification_report(test_trues, test_preds, target_names=EMOS, digits=4, labels=label)
        print(cr)
        with open(save_path + '/CM_%d.txt' % (t), 'w') as f:
            f.write(cr)

if __name__ == '__main__':
    Batch_size = 64
    loss_fc = nn.CrossEntropyLoss().to(device)
    Epochs = 5
    n = 50
    lr = 0.0002
    list = ['bloodmnist']
    for i in range(len(list)):
        print('训练和测试数据集：%s'%(list[i]))
        main(list[i], n)