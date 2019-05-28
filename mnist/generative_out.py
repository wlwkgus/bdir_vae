import torch as t
import torch.nn as nn


class GenerativeOut(nn.Module):
    def __init__(self, fc):
        super(GenerativeOut, self).__init__()

        self.fc = fc

    def forward(self, latent_input, determenistic_input, font_vec, char_vec, transform_vec):
        input = t.cat([latent_input, determenistic_input, font_vec, char_vec, transform_vec], 1)

        return self.fc(input)
