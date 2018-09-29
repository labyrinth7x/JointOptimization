import torchvision as tv
import numpy as np
from PIL import Image


def get_dataset(args):
    # prepare datasets
    cifar10_train_val = tv.datasets.CIFAR10(args.train_root, train=True, download=args.download)
    cifar10_test = tv.datasets.CIFAR10(args.test_root, train=False, download=args.download)

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(cifar10_train_val, args.ratio)
    train = Cifar10Train(args.train_root, train_indexes, train=True, transform=args.transform,)

    if args.dataset_type == 'sym_noise':
        train.symmetric_noise()
    elif args.dataset_type == 'asym_noise':
        train.asymetric_noise()
    else:
        train.pseudo_labels()

    val = Cifar10Val(args.train_root, train_indexes, train=True, transform=args.transform)
    test = Cifar10Val(args.test_root, cifar10_test.test_labels[:], train=False, download=args.download)

    return train, val, test


def train_val_split(train_val, ratio):
    # select {ratio *len(labels)} images from the images.
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    train_num = int(len(train_val) * ratio / 10)

    for id in range(10):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        train_indexes.extend(indexes[:train_num])
        val_indexes.extend(indexes[train_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar10Train(tv.datasets.CIFAR10):
    # including hard labels & soft labels
    def __init__(self, args, root, train_indexes, train=True, transform=None, target_transform=None, download=False):
        super(Cifar10Train, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        self.train_labels = self.train_labels[train_indexes]
        self.train_data = np.array(self.train_data)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), 10), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.train_data), 10) ,dtype=np.int32)
        self._num = int(len(self.train_labels) * self.noise_ratio)
        self._count = 0

    def symmetric_noise(self):
        # to be more equal, every category can be processed separately
        idxes = np.random.permutation(len(self.train_labels))
        for i in range(self.num):
            # train_labels[idxes[i]] -> another category
            label_sym = np.random.randint(10, dtype=np.int32)
            self.train_labels[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][label_sym] = 1

    def asymmetric_noise(self):
        # [airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # truck(9) -> automobile(1)
        # bird(2) -> airplane(0)
        # cat(3) -> dog(5)
        # dog(5) -> cat(3)
        # deer(4) -> horse(7)
        dic = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
        for key in dic.keys():
            idxes = np.where(self.train_labels == key)[0]
            np.random.shuffle(idxes)
            for j in range(self.num):
                self.train_labels[idxes[j]] = dic[key]
                self.soft_labels[idxes[j]][dic[key]] = 1

    def update_labels(self, result):
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.

        idx = self._count % 10
        self.prediction[idx,:] = result

        if self._count >= self.args.begin:
            self.soft_labels = self.prediction.mean(axis = 0)
            # check the paper for this, take the average output prob as s used both in soft and hard labels
            self.labels = self.soft_labels.argmax(axis = 1).astype(np.int32)

        # save params
        if self._count == self.args.epoch:
            np.savez(self.args.dst, hard_labels=self.labels, soft_labels=self.soft_labels)

        self._count += 1

    def reload_labels(self):
        param = np.load(self.args.dst)
        self.labels = param['hard_labels']
        self.soft_labels = param['soft_labels']

    def __getitem__(self, index):
        img, labels, soft_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index]

        # doing this so that it is consistent with all other datasets.
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels, soft_labels, index


# val&test set can be reused.
class Cifar10Val(tv.datasets.CIFAR10):
    def __init__(self, root, val_indexes, train=True, transform=None, target_transform=None, download=False):
        super(Cifar10Val, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # self.train_labels & self.train_data are the attrs from tv.datasets.CIFAR10
        self.val_labels = self.train_labels[val_indexes]
        self.val_data = self.train_data[val_indexes]