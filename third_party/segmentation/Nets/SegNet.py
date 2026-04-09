import torch.nn as nn
import torch
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, label_nbr, drop_path_rate=0.1):
        super(SegNet, self).__init__()
        input_nbr = 3
        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(drop_path_rate)

        # self.down1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        # self.down2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        # self.down3 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        # self.down4 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        # self.down5 = nn.Conv2d(512, 512, kernel_size=2, stride=2)

        # self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)


    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(self.dropout(x12), kernel_size=2, stride=2, return_indices=True)
        # x1p, id1 = self.dropout(self.down1(x12))
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        # x2p, id2 = self.down2(self.dropout(x22))
        x2p, id2 = F.max_pool2d(self.dropout(x22), kernel_size=2, stride=2, return_indices=True)
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(self.dropout(x33), kernel_size=2, stride=2, return_indices=True)
        # x3p, id3 = self.down3(self.dropout(x33))
        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(self.dropout(x43), kernel_size=2, stride=2, return_indices=True)
        # x4p, id4 = self.down4(self.dropout(x43))
        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(self.dropout(x53), kernel_size=2, stride=2, return_indices=True)
        # x5p, id5 = self.down5(self.dropout(x53))

        # Stage 5d
        # F.Upsample()
        diffY = torch.tensor([id5.size()[2] - x5p.size()[2]])
        diffX = torch.tensor([id5.size()[3] - x5p.size()[3]])
        x5p = F.pad(x5p, [0, diffX,
                          0, diffY])
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        # x5d = self.up1(x5p)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        diffY = torch.tensor([id4.size()[2] - x51d.size()[2]])
        diffX = torch.tensor([id4.size()[3] - x51d.size()[3]])

        x51d = F.pad(x51d, [0, diffX,
                            0, diffY])
        # x4d = self.up2(x51d)
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        diffY = torch.tensor([id3.size()[2] - x41d.size()[2]])
        diffX = torch.tensor([id3.size()[3] - x41d.size()[3]])
        x41d = F.pad(x41d, [0, diffX,
                            0, diffY, 0, 0])
        x3d = F.max_unpool2d(self.dropout(x41d), id3, kernel_size=2, stride=2)
        # x3d = self.up3(self.dropout(x41d))
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        diffY = torch.tensor([id2.size()[2] - x31d.size()[2]])
        diffX = torch.tensor([id2.size()[3] - x31d.size()[3]])
        x31d = F.pad(x31d, [0, diffX,
                            0, diffY])
        # x2d = self.up4(self.dropout(x31d))
        x2d = F.max_unpool2d(self.dropout(x31d), id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        diffY = torch.tensor([id1.size()[2] - x21d.size()[2]])
        diffX = torch.tensor([id1.size()[3] - x21d.size()[3]])

        x21d = F.pad(x21d, [0, diffX,
                            0, diffY])
        x1d = F.max_unpool2d(self.dropout(x21d), id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()  # create a copy of the state dict
        th = torch.load(model_path).state_dict()  # load the weigths
        # for name in th:
        # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    torch.backends.cudnn.enabled = True
    print(torch.__version__)
    net = SegNet(2).cuda()
    # net = swin_base_patch4_window12_384().cuda()
    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)
    # 模拟输入数据，例如一批图像
    batch_size = 2  # 每批处理的图像数量
    num_batches = 80  # 总共处理的批次数量
    input_data = torch.randn(batch_size, 3, 800, 600).cuda()  # 模拟输入图像数据


    # 模拟一个简单的前向传播
    def process_image(image):
        return net(image)


    # 将模型切换到评估模式
    net.eval()
    # 计时开始
    start_time = time.time()
    # 模拟处理多批图像
    with torch.no_grad():
        for batch in range(num_batches):
            output = process_image(input_data)  # 调用模型处理图像
    # 计时结束
    end_time = time.time()
    # 计算吞吐量（每秒处理的图像数量）
    total_images_processed = batch_size * num_batches
    throughput = total_images_processed / (end_time - start_time)
    print(f"模型吞吐量: {throughput:.2f} 图像/秒")
