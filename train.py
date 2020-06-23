import models
import torch

from torch import nn, optim
from data import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable
from torchsummary import summary

num_classes = 2
use_cuda = False
def train():
    img_dir = '/opt/data/DogCat/train'
    model = getattr(models, 'ResNet34')(num_classes)
    print(summary(model, (3, 224, 224), device='cpu'))
    print('*'*30)
    train_data = DogCat(img_dir, train=True)
    val_data = DogCat(img_dir, train=False)
    train_dataloader = DataLoader(train_data, 32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, 16, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    lr = 5e-3
    weight_decay = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    for epoch in range(20):
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            target = Variable(label)
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            if ii%100 == 0:
                print('Epoch:{},loss:{}'.format(epoch, loss_meter.value()[0]))

    val_cm, val_accuracy = val(model, val_dataloader)
    print("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm},val_acc:{val_acc}".format(
                epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
                train_cm=str(confusion_matrix.value()), lr=lr, val_acc=val_accuracy))


def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    with torch.no_grad():
        for ii, (input, label) in enumerate(dataloader):
            val_input = Variable(input)
            val_label = Variable(label.type(torch.LongTensor))
            if use_cuda:
                val_input = val_input.cuda()
                val_label = val_label.cuda()
            score = model(val_input)
            confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

            model.train()
            cm_value = confusion_matrix.value()
            right_value = 0
            for i in range(num_classes):
                right_value += cm_value[i][i]
            accuracy = 100*right_value/cm_value.sum()
            return confusion_matrix, accuracy

if __name__ == '__main__':
    train()
