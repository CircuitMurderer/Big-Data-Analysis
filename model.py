import paddle
import paddle.nn as nn
import paddle.nn.functional as f
from sklearn.metrics import accuracy_score, f1_score


class TextCNN(nn.Layer):
    def __init__(self, config, embedding):
        super(TextCNN, self).__init__()
        self.embedding = embedding

        self.convs = nn.LayerList(
            [nn.Conv2D(1, config.filters, (k, self.embedding.embedding_dim)) for k in (2, 3, 4)]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.filters * 3, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.classes)
        )

    @staticmethod
    def conv_and_pool(x, conv):
        x = f.relu(conv(x).squeeze(3))
        x = f.max_pool1d(x, x.shape[2]).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = paddle.concat([self.conv_and_pool(x, conv) for conv in self.convs], axis=1)
        x = self.fc(x)
        return x


def train(model, config, optimizer, criterion, train_loader, dev_loader):
    print('Training...')
    for ep in range(config.epochs):
        all_ind = len(train_loader)
        for ind, (ids, labels) in enumerate(train_loader):
            optimizer.clear_grad()

            ids = paddle.to_tensor(ids)
            labels = paddle.to_tensor(labels)

            preds = model(ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            now = (ind + 1) * 10 // all_ind
            print('\r{}{}%'.format('⬛' * now, now * 10), end='')

        acc, f1 = evaluate(model, dev_loader)
        print('\nEpoch:', ep + 1, 'Accuracy:', acc, 'F1:', f1)


def evaluate(model, dev_loader):
    total_loss = 0.
    total_acc = 0.
    total_f1 = 0.

    model.eval()
    for ind, (ids, labels) in enumerate(dev_loader):
        logits = model(ids)
        preds = paddle.argmax(f.softmax(logits, axis=1), axis=1)

        y_pred = preds.flatten().numpy()
        y_true = labels.flatten().numpy()

        acc = accuracy_score(y_pred, y_true)
        f1 = f1_score(y_pred, y_true)

        total_acc += acc
        total_f1 += f1

    model.train()
    return total_acc / len(dev_loader), total_f1 / len(dev_loader)
