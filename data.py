import glob
import random
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms_=None, mode='train'):
        transforms_ = [transforms.Resize(int(143), Image.BICUBIC),
                       transforms.RandomCrop(128),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transforms_)
        # content source
        self.X = []
        content_source = sorted(glob.glob(os.path.join(content_dir, mode, '*')))
        for label, content in enumerate(content_source):
            if content.endswith('.jpg'):
                self.X.append(content)
            else:
                for path in sorted(glob.glob(content_source[label] + "/*")):
                    self.X.append(path)

        # style image source(s)
        self.Y = []
        style_sources = sorted(glob.glob(os.path.join(style_dir, mode, '*')))
        for label, style in enumerate(style_sources):
            print('label %d: %s' % (label, style))
            temp = [(label, x) for x in sorted(glob.glob(style_sources[label] + "/*"))]
            self.Y.extend(temp)

        print(len(self.X))

    def __getitem__(self, index):
        output = {'content': self.transform(Image.open(self.X[index % len(self.X)]).convert('RGB'))}

        # select style
        selection = self.Y[random.randint(0, len(self.Y) - 1)]

        try:
            output['style'] = self.transform(Image.open(selection[1]).convert('RGB'))
        except:
            print(selection)

        output['style_label'] = selection[0]

        return output

    def __len__(self):
        return max(len(self.X), len(self.Y))


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


if __name__ == '__main__':
    from tqdm import tqdm
    # content_dir = '../../places365'
    # style_dir = '../../wikiart'

    content_dir = '../../photo2fourcollection/content'
    style_dir = '../../photo2fourcollection/style'

    train_dataset = ImageDataset(content_dir, style_dir, mode='train')
    print('train_dataset size:', len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    # for i, batch in tqdm(enumerate(train_loader, 1)):
    #     pass
    #     print(batch['content'].size())
    #     print(batch['style'].size())
    #     print(batch['style_label'].size())