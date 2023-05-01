import torch
from torch.autograd.variable import Variable
from torchvision.utils import save_image


def train_one_epoch(epoch,epochs,i,train_loader,real_images,optim_D,latent_dim,discriminator,criterion,generator,optim_G):
    batch_size = real_images.size(0)

    # 真实和虚假标签
    real_labels = torch.ones(batch_size, 1).cuda()
    fake_labels = torch.zeros(batch_size, 1).cuda()

    # 训练判别器
    optim_D.zero_grad()
    real_outputs = discriminator(real_images)
    real_loss = criterion(real_outputs, real_labels)

    z = Variable(torch.randn(batch_size, latent_dim)).cuda()
    fake_images = generator(z)
    fake_outputs = discriminator(fake_images)
    fake_loss = criterion(fake_outputs, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optim_D.step()

    # 训练生成器
    optim_G.zero_grad()
    z = Variable(torch.randn(batch_size, latent_dim)).cuda()
    fake_images = generator(z)
    fake_outputs = discriminator(fake_images)
    # 使用相反的标签来优化生成器
    g_loss = criterion(fake_outputs, real_labels)
    g_loss.backward()
    optim_G.step()
    
    #输出损失信息
    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    # 保存生成器生成的图片
    if (epoch+1) % 10 == 0:
        save_image(fake_images.view(fake_images.size(0), 1, 28, 28), f'images/epoch_{epoch+1}.png')
    
    
