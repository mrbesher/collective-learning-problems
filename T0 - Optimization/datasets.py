from math import ceil

import torch

from torchvision import datasets as vision_datasets, transforms
from torchtext import datasets as text_datasets

from torch.utils.data import DataLoader

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

dataset_features = {
    'imdb': {
        'vocab_len': 0,
        'n_classes': 0
    },
    'ag_news': {
        'vocab_len': 0,
        'n_classes': 0
    }
}


# Classes for text
class TensorTuple():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
        return (self.x.to(device), self.y.to(device))


class TextDataLoader(DataLoader):
    def __init__(self, data_iterator, n_items, batch_size=8, shuffle=False, collate_fn=None):
        super().__init__(data_iterator, batch_size=batch_size,
                         shuffle=shuffle, collate_fn=collate_fn)
        self.n_items = ceil(n_items / batch_size)

    def __len__(self):
        return self.n_items


def yield_tokens(tokenizer, data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def collate_batch(batch, text_pipeline, label_pipeline):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return TensorTuple(text_list, offsets), label_list

# Getters for text datasets
# Reference: https://colab.research.google.com/drive/1WUy4G2SsoLelrZDkO2I0v9tHx9x27NJK?usp=sharing
def get_imdb_data(batch_size=8):
    train_iter, test_iter = text_datasets.IMDB()

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(
        tokenizer, train_iter), min_freq=20, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    classes = set(label for label, _ in train_iter)

    dataset_features['imdb']['vocab_len'] = len(vocab)
    dataset_features['imdb']['n_classes'] = len(classes)
    encoder_dict = {k: i for i, k in enumerate(classes)}

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: encoder_dict.get(x)

    train_len = len([1 for _ in train_iter])
    test_len = len([1 for _ in test_iter])

    train_dataloader = TextDataLoader(train_iter, train_len, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline))
    test_dataloader = TextDataLoader(train_iter, test_len, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline))

    return train_dataloader, test_dataloader

def get_agnews_data(batch_size=8):
    train_iter, test_iter = text_datasets.IMDB()

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(
        tokenizer, train_iter), min_freq=20, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    classes = set(label for label, _ in train_iter)

    dataset_features['ag_news']['vocab_len'] = len(vocab)
    dataset_features['ag_news']['n_classes'] = len(classes)
    encoder_dict = {k: i for i, k in enumerate(classes)}

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: encoder_dict.get(x)

    train_len = len([1 for _ in train_iter])
    test_len = len([1 for _ in test_iter])

    train_dataloader = TextDataLoader(train_iter, train_len, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline))
    test_dataloader = TextDataLoader(train_iter, test_len, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline))

    return train_dataloader, test_dataloader


# Generic transform for all image datasets
transformers = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_cifar_data(batch_size=128, data_path='data'):
    train_loader = DataLoader(
        vision_datasets.CIFAR10(data_path, train=True,
                                download=True, transform=transformers),
        batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        vision_datasets.CIFAR10(data_path, train=False,
                                transform=transformers),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_mnist_data(batch_size=128, data_path='data'):
    train_loader = DataLoader(
        vision_datasets.MNIST(data_path, train=True,
                              download=True, transform=transformers),
        batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        vision_datasets.MNIST(data_path, train=False, transform=transformers),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# A dictionary of functions to initialize datasets
data_loading_functions = {
    'cifar': get_cifar_data,
    'mnist': get_mnist_data,
    'imdb': get_imdb_data,
    'ag_news': get_agnews_data
}


def get_data(dataset_name, batch_size=128, data_path='data'):
    getter = data_loading_functions[dataset_name]
    return getter(batch_size=batch_size, data_path=data_path)
