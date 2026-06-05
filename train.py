import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from thop import profile


from  ShuffleNet import ShuffleNetV2
from mydataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate

# ==========================================

# ==========================================
MANUAL_LAMBDA = 0.9
MANUAL_BETA = 0.0001


# ==========================================

class EMAFocalLoss(nn.Module):
    """

    """

    def __init__(self, model, num_classes=6, gamma=2.0, lambda_ema=0.9, beta=0.0001, device='cuda'):
        super(EMAFocalLoss, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_ema = lambda_ema
        self.beta = beta
        self.device = device


        self.register_buffer('alpha', torch.ones(num_classes).to(device))

        self.register_buffer('class_acc_ema', torch.ones(num_classes).to(device) * 0.5)

    def update_alpha(self, current_class_acc):
        """

        """

        self.class_acc_ema = self.lambda_ema * self.class_acc_ema + \
                             (1 - self.lambda_ema) * current_class_acc.to(self.device)


        target_weight = torch.where(self.class_acc_ema < 0.5,
                                    torch.tensor(2.0).to(self.device),
                                    torch.tensor(1.0).to(self.device))


        self.alpha = self.lambda_ema * self.alpha + (1 - self.lambda_ema) * target_weight

    def forward(self, inputs, targets):


        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)


        batch_alpha = self.alpha[targets]


        focal_loss = batch_alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss_mean = focal_loss.mean()


        l2_reg = torch.tensor(0., device=self.device)
        if self.beta > 0:
            for param in self.model.parameters():
                if param.requires_grad:
                    l2_reg += torch.sum(torch.pow(param, 2))


        total_loss = focal_loss_mean + self.beta * l2_reg

        return total_loss


@torch.no_grad()
def get_class_accuracy(model, data_loader, device, num_classes):
    """

    """
    model.eval()
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)

        for c in range(num_classes):
            indices = (labels == c)
            if indices.sum() > 0:
                class_total[c] += indices.sum().item()
                class_correct[c] += (pred[indices] == c).sum().item()


    class_acc = class_correct / (class_total + 1e-6)
    return class_acc


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    tb_writer = SummaryWriter()


    save_path = f"./weights"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = ShuffleNetV2(num_classes=args.num_classes).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    input_size = (3, 224, 224)
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


    criterion = EMAFocalLoss(
        model=model,
        num_classes=args.num_classes,
        lambda_ema=MANUAL_LAMBDA,
        beta=MANUAL_BETA,
        device=device
    )
    print(f"Using EMAFocalLoss with Lambda={MANUAL_LAMBDA}, Beta={MANUAL_BETA}")

    best_acc = 0.0


    for epoch in range(args.epochs):
        epoch_start_time = time.time()


        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    criterion=criterion)


        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)


        class_acc_tensor = get_class_accuracy(model, val_loader, device, args.num_classes)
        criterion.update_alpha(class_acc_tensor)

        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch}] Updated Alphas: {criterion.alpha.cpu().numpy()}")

        scheduler.step(mean_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        train_throughput = len(train_loader.dataset) / epoch_duration

        print(
            f"[epoch {epoch}] accuracy: {round(acc, 3)}, time: {epoch_duration:.2f}s, Train Throughput: {train_throughput:.1f} img/s")

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)


        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"{save_path}/model-{epoch + 1}.pth")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")

    print(f"Training completed for Beta={MANUAL_BETA}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default=r"")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)