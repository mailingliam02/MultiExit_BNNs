
import torch
from torchvision import transforms, datasets
def get_test_data(hyperparams):
    mean=[0.5071, 0.4865, 0.4409]
    std=[0.2673, 0.2564, 0.2762]
    normalize = transforms.Normalize(mean = mean,std = std)   
    train_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    train_set = datasets.CIFAR100(
                root="data/cifar100", train=False,
                download=False, transform=train_transforms,
            )
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 1,
                num_workers=1, pin_memory=True)
    single_item_v2 = next(iter(train_loader))
    return single_item_v2, train_loader