import torch
from torch import nn, optim
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
triplet_loss = nn.TripletMarginLoss()

# netにはモデルを代入（とりあえず文字列）
net = "model"

for epoch in range(n_iter):
    running_loss = 0.0
    net.train()
    n = 0
    n_acc = 0
    # xxにテキスト、yyに画像?
    for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        xx = xx.to(device)
        yy = yy.to(device)

        anchor = torch.randn(100, 128, requires_grad=True)
        positive = torch.randn(100, 128, requires_grad=True)
        negative = torch.randn(100, 128, requires_grad=True)
        output = triplet_loss(anchor, positive, negative)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        running_loss += output.item()
        n += len(xx)
    print(epoch)


