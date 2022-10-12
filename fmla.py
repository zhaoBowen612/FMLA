import torch
from torch import nn
from torch.nn import functional as F

from position import PositionalEncoding
from involution import Involution
from deformable import Deformable
from drop import PointwiseDropout

import logging
import args


class FMLA(nn.Module):
    def __init__(self, sequence_in, num_classes, head=args.head, etc=args.etc, drop=args.drop):
        super(FMLA, self).__init__()
        print(head, etc, drop)
        self.head_dim = [128 // head, 128 // head, 64 // head, 64 // head]
        self.num_classes = num_classes
        self.embed = nn.Conv1d(1, 128, 3, padding=3 // 2)
        self.pe = PositionalEncoding(128, sequence_in)

        self.b1 = Block(128, 128, head, etc, sequence_in)
        self.ff1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.ff12 = nn.Conv1d(128, 128, 1)
        self.bn12 = nn.BatchNorm1d(128)
        self.res1 = ResBlock(128, 128, 3, sequence_in)
        self.e1 = nn.ModuleList(nn.Conv1d(128 // head, etc, 1) for i in range(head))

        self.b2 = Block(128, 128, head, etc, sequence_in)
        self.ff2 = nn.Conv1d(128, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.ff22 = nn.Conv1d(128, 128, 1)
        self.bn22 = nn.BatchNorm1d(128)
        self.res2 = ResBlock(128, 128, 3, sequence_in)
        self.e2 = nn.ModuleList(nn.Conv1d(128 // head, etc, 1) for i in range(head))

        self.b3 = Block(128, 64, head, etc, sequence_in)
        self.ff3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.ff32 = nn.Conv1d(64, 64, 1)
        self.bn32 = nn.BatchNorm1d(64)
        self.res3 = ResBlock(128, 64, 3, sequence_in)
        self.res32 = nn.Conv1d(128, 64, 1)
        self.e3 = nn.ModuleList(nn.Conv1d(64 // head, etc, 1) for i in range(head))

        self.b4 = Block(64, 64, head, etc, sequence_in)
        self.ff4 = nn.Conv1d(64, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.ff42 = nn.Conv1d(64, 1, 1)
        self.bn42 = nn.BatchNorm1d(1)
        self.res4 = ResBlock(64, 64, 3, sequence_in)
        self.res42 = nn.Conv1d(64, 1, 1)
        self.e4 = nn.ModuleList(nn.Conv1d(64 // head, etc, 1) for i in range(head))

        self.squeeze = nn.Conv1d(64, 1, 1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1 * sequence_in, num_classes)
        self.pap = nn.AdaptiveAvgPool1d(1)
        self.gap = nn.Sequential(nn.Conv1d(64, num_classes, 1), nn.BatchNorm1d(num_classes), nn.AdaptiveAvgPool1d(1))
        self.weighted = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, mask=1, regular=False):
        x = self.embed(x)
        y_pe = self.pe(x)

        x1 = self.res1(x)
        e1s = [e(x1[:, i * self.head_dim[0]:(i + 1) * self.head_dim[0], :]) for i, e in enumerate(self.e1)]
        y = self.bn1(self.b1(y_pe, e1s, regular, mask) + y_pe)
        y1 = self.bn12(self.ff12(F.gelu(self.ff1(y))) + y)

        x2 = self.res2(x1)
        e2s = [e(x2[:, i * self.head_dim[1]:(i + 1) * self.head_dim[1], :]) for i, e in enumerate(self.e2)]
        y = self.bn2(self.b2(y1, e2s, regular, mask) + y1)
        y2 = self.bn22(self.ff22(F.gelu(self.ff2(y))) + y)

        x3 = self.res3(x2)
        e3s = [e(x3[:, i * self.head_dim[2]:(i + 1) * self.head_dim[2], :]) for i, e in enumerate(self.e3)]
        y = self.bn3(self.b3(y2, e3s, regular, mask) + self.res32(y2))
        y3 = self.bn32(self.ff32(F.gelu(self.ff3(y))) + y)

        x4 = self.res4(x3)
        e4s = [e(x4[:, i * self.head_dim[3]:(i + 1) * self.head_dim[3], :]) for i, e in enumerate(self.e4)]
        y = self.bn4(self.b4(y3, e4s, regular, mask) + y3)
        y4 = self.bn42(self.ff42(F.gelu(self.ff4(y))) + self.res42(y))

        y = self.dense(self.flatten(y4))
        x = self.gap(x4).squeeze(-1)
        return F.log_softmax(x + y, dim=-1), F.log_softmax(x / 3, dim=-1), F.log_softmax(y / 3, dim=-1), [x1, x2, x3], [
            y1, y2, y3]


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, head, etc_size, length):
        super(Block, self).__init__()
        self.head = head
        self.etc = etc_size
        self.dim_out = dim_out
        self.sqrt = nn.Parameter(torch.tensor(128.))
        self.to_q = nn.Conv1d(dim_in, dim_out, 1)
        # self.to_k = nn.Conv1d(dim_in, dim_out, 1)
        self.trans_k = nn.Conv1d(dim_out // head, dim_out // head, 1)
        self.to_v = nn.Conv1d(dim_in, dim_out * head, 1)
        # self.trans_v = nn.ModuleList(nn.Conv1d(dim_out // head, dim_out // head, kernel_size=1) for i in range(head))
        # self.trans_v = nn.ModuleList(nn.Conv1d(length, etc_size, 1) for i in range(head))  # for shorten

        self.reatten = nn.Conv1d(dim_out * head, dim_out, kernel_size=1)
        self.mix = Mixing(dim_out, head)
        self.drop = PointwiseDropout(args.drop, length)
        self.squeeze = nn.Conv1d(dim_out * head, dim_out, 1)

    def forward(self, y, e_s, regular, mask):
        bat, dim_out, seq = y.size()
        multi_q = self.to_q(y)[:, None] * self.mix()[None]
        for h in range(self.head):
            multi_q[:, h] = self.drop(multi_q[:, h], mask, regular)
        multi_v = self.to_v(y)
        # multi_v = self.drop(self.to_v(y), mask, regular)
        for h in range(self.head):
            multi_v[:, h * int(dim_out / self.head):(h+1) * int(dim_out / self.head)] = self.drop(
                multi_v[:, h * int(dim_out / self.head):(h+1) * int(dim_out/self.head)], mask, regular)

        etc_v = torch.cat([torch.matmul(multi_v[:, h * self.dim_out:(h + 1) * self.dim_out, :],
                                        e_s[h].transpose(1, 2)) for h in range(self.head)], dim=1)
        etc_k = self.reatten(etc_v.view(bat, self.head * self.dim_out, -1))

        multi_atten = torch.softmax(torch.matmul(etc_k[:, None].transpose(-1, -2), multi_q) / self.sqrt, dim=-2)
        out = torch.matmul(etc_v.view(bat, self.head, -1, self.etc), multi_atten).reshape(bat, -1, seq) + \
              F.avg_pool1d(multi_q.reshape(bat, -1, seq), 3, 1, 3 // 2)
        out = self.squeeze(out.view(bat, -1, seq))
        return out


class Mixing(nn.Module):
    def __init__(self, dim_out, num_head):
        super(Mixing, self).__init__()
        self.mix = nn.Parameter(torch.randn(num_head, dim_out, 1))

    def forward(self):
        return self.mix


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, sequence_in):
        super(ResBlock, self).__init__()
        self.dconv1 = Deformable(dim_in, dim_out, kernel_size, sequence_in)
        self.dconv2 = Deformable(dim_out, dim_out, kernel_size, sequence_in)
        self.dconv3 = Deformable(dim_out, dim_out, kernel_size, sequence_in)
        self.res = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else None

    def forward(self, x):
        out = self.dconv1(x)
        out = self.dconv2(out)
        out = self.dconv3(out)
        out += self.res(x) if self.res else x
        return out
