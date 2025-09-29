import random
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from mnist_loader import show_images, MnistDataloader


class FeedForward(nn.Module):
    def __init__(self, resolution, hidden_size=16, hidden_layers=2, dropout=0.1):
        super(FeedForward, self).__init__()
        self.input_ff = nn.Linear(resolution * resolution, hidden_size)
        self.ff_list = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers - 1)])
        self.output_ff = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 28, 28] or [B, 784]
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.input_ff(x)))
        for ff in self.ff_list:
            x = self.dropout(F.relu(ff(x)))

        return self.output_ff(x)


class HandwrittenDigitsClassifier(nn.Module):
    def __init__(self, resolution, hidden_size=16, hidden_layers=2, dropout=0.1):
        super(HandwrittenDigitsClassifier, self).__init__()
        self.ff = FeedForward(resolution, hidden_size, hidden_layers, dropout)

    def forward(self, x):
        return self.ff(x)


def _show_random_images(x_train, y_train, x_test, y_test):
    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)


def main():
    input_path = "mnist-dataset"
    training_images_filepath = join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte")
    training_labels_filepath = join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
    mnist_loader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

    # 1） 转化为张量，并归一化到0-1之间
    # 为什么输入要归一化？ 
    # 因为输入的像素值是0-255，归一化后可以使得输入的像素值在0-1之间，这样可以使得模型更容易收敛
    x_train = torch.tensor(x_train, dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 2）DataLoader
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=True)

    # 3）模型/损失/优化器/设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenDigitsClassifier(28).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4）训练 评估
    epochs = 12
    for epoch in range(epochs):
        # train
        train_acc, train_loss = model_train(criterion, device, model, optimizer, train_dl)

        # eval
        test_acc, test_loss = model_eval(criterion, device, model, test_dl)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    # torch.save(model.state_dict(), "mnist_ffn.pth")

    # 测试推理
    (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
    N = 3
    images = random.sample(x_test, N)
    show_images(images, ["" for _ in range(N)])
    images = [(torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0).to(device) for image in images]

    model.eval()
    with torch.no_grad():
        for image in images:
            logits = model(image)
            pred = logits.argmax(dim=1).item()
            print(pred)


def model_eval(criterion, device, model, test_dl):
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            eval_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    test_loss = eval_loss / total
    test_acc = correct / total
    return test_acc, test_loss


def model_train(criterion, device, model, optimizer, train_dl):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, target in train_dl:
        inputs, targets = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    return train_acc, train_loss


if __name__ == '__main__':
    main()
