import torch.nn as nn


class TextNet(nn.Module):
    def __init__(self, n_input, n_hidden, layers):
        super(TextNet, self).__init__()

        self.input_layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Dropout(0.1),
            # nn.ReLU()
        )

        mid_layers = []
        for _ in range(layers):
            mid_layers.append(nn.Linear(n_hidden, n_hidden))
            mid_layers.append(nn.Dropout(0.1))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.out_layres = nn.Sequential(
            nn.Linear(n_hidden, int(n_hidden / 2)),
            nn.Sigmoid(),  # nn.ReLU(), #
            nn.Linear(int(n_hidden / 2), 1)
        )

    def forward(self, x):
        y = self.input_layers(x)
        y = self.mid_layers(y)
        y = self.out_layres(y)
        return y