from model import Generator,Discriminator
from train import train_one_epoch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 超参数设置
batch_size = 666
epochs = 100
lr = 0.0002
latent_dim = 100

# 图像转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载MNIST数据集
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 创建生成器和判别器
generator = Generator(latent_dim).cuda()
discriminator = Discriminator().cuda()

# 优化器
optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_D = optim.Adam(discriminator.parameters(), lr=lr)

# 损失函数
criterion = nn.BCELoss().cuda()

# 训练
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.cuda()
        train_one_epoch(epoch = epoch,
                        epochs= epochs,
                        i = i,
                        train_loader= train_loader,
                        real_images= real_images,
                        optim_D=optim_D,
                        latent_dim=latent_dim,
                        discriminator=discriminator,
                        criterion=criterion,
                        generator=generator,
                        optim_G=optim_G,
                        )
