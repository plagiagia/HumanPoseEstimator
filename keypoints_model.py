import torch.nn as nn

from calculate_accuracy import *


# Build the network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.checkpoint_path = 'model_checkpoints'

        # 1st CONVOLUTION, IN_CHANNELS = 3 (RGB), OUT_CHANNELS=64, KERNEL=7x7, STRIDE=2, PADDING=3
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        nn.init.xavier_normal_(self.Conv1.weight)
        nn.init.constant_(self.Conv1.bias, 0)
        # END 1st CONVOLUTION, OUT_CONV_SHAPE = [(192−7+2*3)/2]+1=[96, 128]

        # 1st SEQUENTIAL LAYER
        self.layer1 = nn.Sequential(
            self.Conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # END 1st SEQUENTIAL, OUT_MAX_POOL_SHAPE = [96, 128] / 2 = [48, 64]

        # 2nd CONVOLUTION, IN_CHANNELS=64, OUT_CHANNELS=128, KERNEL=5x5, STRIDE=1, PADDING=2
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_normal_(self.Conv2.weight)
        nn.init.constant_(self.Conv2.bias, 0)
        # END 2nd CONVOLUTION, OUT_CONV_SHAPE = [(48−5+2*2)/1]+1=[48, 64]

        # 2nd SEQUENTIAL LAYER
        self.layer2 = nn.Sequential(
            self.Conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # END 2nd SEQUENTIAL, OUT_MAX_POOL_SHAPE = [48, 64] / 2 = [24, 32]

        # 3rd CONVOLUTION, IN_CHANNELS=128, OUT_CHANNELS=256, KERNEL=5x5, STRIDE=1, PADDING=2
        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_normal_(self.Conv3.weight)
        nn.init.constant_(self.Conv3.bias, 0)
        # END 3rd CONVOLUTION, OUT_CONV_SHAPE = [(24−5+2*2)/1]+1=[24, 32]

        # 3rd SEQUENTIAL LAYER
        self.layer3 = nn.Sequential(
            self.Conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # END 2nd SEQUENTIAL, OUT_MAX_POOL_SHAPE = [24, 32] / 2 = [12, 16]

        # Deconvolution
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, padding=1, output_padding=0, kernel_size=4,
                                          stride=2)
        nn.init.xavier_normal_(self.deconv4.weight)
        self.layer4 = nn.Sequential(
            self.deconv4,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Deconvolution
        self.deconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, padding=1, output_padding=0, kernel_size=4,
                                          stride=2)
        nn.init.xavier_normal_(self.deconv5.weight)
        self.layer5 = nn.Sequential(
            self.deconv5,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # FINAL CONVOLUTION, IN_CHANNELS=256, OUT_CHANNELS=17, KERNEL=1x1, STRIDE=1, PADDING=0
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=17, kernel_size=1)
        nn.init.xavier_normal_(self.final_conv.weight)
        nn.init.constant_(self.final_conv.bias, 0)
        # END FINAL CONVOLUTION, OUT_CONV_SHAPE = [(24−1+2*0)/1]+1=[24, 32]

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.final_conv(out)

        return out

    # Loss function
    @staticmethod
    def custom_l2_loss_closure(mse_loss_fn):
        # We want to zero out the loss for when no label was provided in the target heatmap
        # We can achieve this by multiply the validity tensor with the pred

        def custom_l2_loss(pred, target, validity):
            v = validity.unsqueeze(2).unsqueeze(2)
            pred = pred * v
            target = target * v
            return mse_loss_fn(pred, target)

        return custom_l2_loss

    # function to store model checkpoint
    def checkpoint_model(self, optimizer, epoch, batch):
        state = {'model': self.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.checkpoint_path + "_posenet_" + str(epoch) + "_" + str(batch) + '.pth')

    def fit(self, epochs, train_dataloder, val_dataloader, optimizer, loss_fn, scheduler):
        for j in range(epochs):
            for i_batch, sample_batched in enumerate(train_dataloder):

                input_img = sample_batched['image'].to('cuda')
                heatmap = sample_batched['heatmap'].to('cuda')
                validity = sample_batched['validity'].to('cuda')

                # Clear out accumulated gradients
                optimizer.zero_grad()

                output = self(input_img)
                loss = loss_fn(output, heatmap, validity)

                loss.backward()
                optimizer.step()

                if i_batch % 500 == 0:
                    acc_train = accuracy(output.detach().cpu().numpy(), heatmap.detach().cpu().numpy())
                    acc_val = get_accuracy(self, val_dataloader)
                    print("Epoch\t", j, "Batch\t", i_batch, "Training Loss: \t", loss.item(), "Train Accuracy: \t",
                          acc_train,
                          "Val Accuracy: \t", acc_val)

                if i_batch % 5000 == 0:
                    self.checkpoint_model(optimizer, j, i_batch)

            scheduler.step()
        self.checkpoint_model(optimizer, j, i_batch)
