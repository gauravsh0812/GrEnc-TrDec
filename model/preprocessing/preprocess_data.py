import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import (Dataset, 
                              DataLoader,
                              DistributedSampler, 
                              SequentialSampler)

from collections import Counter
from torchtext.vocab import Vocab

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.dataframe = dataframe
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        eqn = self.dataframe.iloc[index, 1]
        indexed_eqn = []
        tokens = self.tokenizer(eqn)
        for token in tokens:
            if self.vocab.stoi[token] is not None:
                indexed_eqn.append(self.vocab.stoi[token])
            else:
                indexed_eqn.append(self.vocab.stoi["<unk>"])

        return (
            self.dataframe.iloc[index, 0], 
            torch.Tensor(indexed_eqn)
        )


class My_pad_collate(object):
    """
    padding mml to max_len, and stacking images
    return: mml_tensors of shape [batch, max_len]
            stacked image_tensors [batch]
    """

    def __init__(self, device, vocab, max_len):
        self.device = device
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.stoi["<pad>"]

    def __call__(self, batch):
        _img, _mml = zip(*batch)

        # padding mml
        # padding to a fix max_len equations with more tokens than
        # max_len will be chopped down to max_length.

        batch_size = len(_mml)
        padded_mml_tensors = (
            torch.ones([batch_size, self.max_len], dtype=torch.long)
            * self.pad_idx
        )
        for b in range(batch_size):
            if len(_mml[b]) <= self.max_len:
                padded_mml_tensors[b][: len(_mml[b])] = _mml[b]
            else:
                padded_mml_tensors[b][: self.max_len] = _mml[b][: self.max_len]

        # graphs and images tensors (they both have same numbering)
        _img = torch.Tensor(_img)

        return (
            _img.to(self.device),
            padded_mml_tensors.to(self.device),
        )


def preprocess_dataset(args):
    print("preprocessing data...")

    # reading raw text files
    mml_path = f'{args["path_to_data"]}/{args["markup"]}.lst'
    mml_txt = open(mml_path).read().split("\n")[:-1]  # [:-1] to avoid the last \n which increaes the length by 1.
    image_num = range(0, len(mml_txt))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        image_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_images in enumerate([train_images, test_images, val_images]):
        raw_mml_data = {
            "IMG": [num for num in t_images],   # both graph and img gonna have same numbering
            "EQUATION": [
                ("<sos> " + mml_txt[num] + " <eos>") for num in t_images
            ],
        }

        if t_idx == 0:
            train = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])
        elif t_idx == 1:
            test = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])
        else:
            val = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])

    # build vocab
    print("building vocab...")

    counter = Counter()
    for line in train["EQUATION"]:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    vocab = Vocab(
        counter,
        min_freq=args["vocab_freq"],
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
    )

    # writing vocab file...
    vfile = open("vocab.txt", "w")
    for vidx, vstr in vocab.stoi.items():
        vfile.write(f"{vidx} \t {vstr} \n")

    # define tokenizer function
    def tokenizer(x):
        return x.split()

    # initializing pad collate class
    mypadcollate = My_pad_collate(args["device"], vocab, args["max_len"])

    print("saving dataset files to data/ folder...")

    train.to_csv(
        f"{args['path_to_data']}/train.csv",
        index=False,
    )
    test.to_csv(
        f"{args['path_to_data']}/test.csv", index=False
    )
    val.to_csv(
        f"{args['path_to_data']}/val.csv", index=False
    )

    print("building dataloaders...")

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train, vocab, tokenizer)
    # creating dataloader
    if args["ddp"]:
        train_sampler = DistributedSampler(
            dataset=imml_train,
            num_replicas=args["world_size"],
            rank=args["rank"],
            shuffle=True,
        )
        sampler = train_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = args["shuffle"]
    train_dataloader = DataLoader(
        imml_train,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=args["pin_memory"],
    )

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val, vocab, tokenizer)

    if args["ddp"]:
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = args["shuffle"]

    val_dataloader = DataLoader(
        imml_val,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=args["pin_memory"],
    )

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test, vocab, tokenizer)
    if args["ddp"]:
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = args["shuffle"]

    test_dataloader = DataLoader(
        imml_test,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=shuffle,
        sampler=None,
        collate_fn=mypadcollate,
        pin_memory=args["pin_memory"],
    )

    return train_dataloader, test_dataloader, val_dataloader, vocab