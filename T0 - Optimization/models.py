from torch import nn
import torch.nn.functional as F

from datasets import dataset_features

dataset_hyperparams = {
    'cifar': {
        'epochs': 2,
        'lr': 1e-3
    },
    'mnist': {
        'epochs': 2,
        'lr': 1e-3
    },
    'imdb': {
        'epochs': 2,
        'lr': 1e-3
    },
    'ag_news': {
        'epochs': 2,
        'lr': 1e-3
    }
}

class MyFC(nn.Module):
  def __init__(self, input_size, output_size, hidden_units=200, dropout_p=0.2):
    """
    Returns a FC network of two layers. With ReLU activation after the first one.
    """

    super(MyFC, self).__init__()
    self.dropout_p = dropout_p
    self.input_size = input_size
    self.fc1 = nn.Linear(input_size, hidden_units)
    self.fc2 = nn.Linear(hidden_units, output_size)
  
  def forward(self, x):
    x = x.view(-1, self.input_size)

    x = self.fc1(x)
    x = F.relu(F.dropout(x, p=self.dropout_p, training=self.training))
    x = self.fc2(x)

    return x

class ConvNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, width=None, height=None, in_channels=3, class_count=10):
        super(ConvNet, self).__init__()

        if (not (width and height)) or (height != width):
          raise Exception(f'Must have valid and equal width and height. Got (WxH)=({width}x{height})')

        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5, padding='valid')
        self.bn1 = nn.BatchNorm2d(20)

        # output_size = (W – F + 2P) / S + 1 [Modified for maxpool]
        width = width - 5 + 1

        self.conv2 = nn.Conv2d(20, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)

        width = width // 2

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)

        width = width // 2
        self.fc = MyFC(width * width * 32, class_count)


    def forward(self, x):
        x = F.dropout2d(self.conv1(x), p=0.2, training=self.training)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.dropout2d(self.conv2(x), p=0.2, training=self.training)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.dropout2d(self.conv3(x), p=0.5, training=self.training)
        x = F.max_pool2d(x, 2)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.fc(x)

        return x


class TextClassificationNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = MyFC(embed_dim, num_class)

    def forward(self, x):
        text, offsets = x
        embeddings = self.embedding(text, offsets)
        return self.fc(embeddings)

init_cifar_model = lambda: ConvNet(32, 32)
init_mnist_model = lambda: ConvNet(28, 28, in_channels=1)

def init_imdb_model():
    vocab_len = dataset_features['imdb']['vocab_len']
    n_classes = dataset_features['imdb']['n_classes']

    if (not vocab_len) or (not n_classes):
        raise Exception("Cannot get vocab len or # of classes. Initialize dataset first")

    return TextClassificationNN(vocab_len, 32, n_classes)

def init_agnews_model():
    vocab_len = dataset_features['ag_news']['vocab_len']
    n_classes = dataset_features['ag_news']['n_classes']

    if (not vocab_len) or (not n_classes):
        raise Exception("Cannot get vocab len or # of classes. Initialize dataset first")

    return TextClassificationNN(vocab_len, 32, n_classes)


model_initializers = {
    'cifar': init_cifar_model,
    'mnist': init_mnist_model,
    'imdb': init_imdb_model,
    'ag_news': init_agnews_model
}

def init_model(dataset_name):
    model_initializer = model_initializers[dataset_name]
    return model_initializer()