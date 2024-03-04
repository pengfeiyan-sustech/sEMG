import os
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder

DATASETS = [
    "PACS",
]


def get_dataset_class(dataset_name):
    """
    :param dataset_name: 数据集名称
    :return: 对应的数据集类
    """
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    """返回数据集类的领域数量"""
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    """
    这个类本身不能直接实例化，因为它是一个抽象基类。
    它只提供了一些默认行为和属性，子类需要根据具体的领域和需求来实现和扩展。
    """
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)

            ImageFolder_with_idx = dataset_with_indices(ImageFolder, i)
            env_dataset = ImageFolder_with_idx(path,
                                               transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


def dataset_with_indices(cls, env_i=None):
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        if env_i == None:
            return data, target, index
        else:
            return data, target, index, env_i

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]


if __name__ == '__main__':
    root = r'D:\StudyFiles\StudyNotes\paperWork\code\testData\PACS'
    pacs_dataset = PACS(root=root, test_envs=[0, 1], augment=False, hparams=None)
    print(pacs_dataset.datasets[2][5])
