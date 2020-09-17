import torch
from torch.utils.data import DataLoader

from keypoints_dataset import AnnotationsDataset
from keypoints_model import ConvNet

if __name__ == "__main__":
    # CONSTANTS
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10

    # MODEL
    model = ConvNet()
    model.cuda()
    model.train()

    # DATASETS
    train_dataset = AnnotationsDataset(annotations_path='./annotations/person_keypoints_train2017.json',
                                       img_dir='./train2017')
    val_dataset = AnnotationsDataset(annotations_path='./annotations/person_keypoints_val2017.json',
                                     img_dir='./val2017')

    # DATA LOADER
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    # LOSS
    loss_fn = model.custom_l2_loss_closure(mse_loss_fn=torch.nn.MSELoss())

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 8], 0.1)

    model.fit(NUM_EPOCHS, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler)
