import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class DifferenceMoE(nn.Module):
    '''
    Difference MoE Module
    param 
    ablation:
        moe: e1+e2
        dmoe: e1-alpha*e2
    expert_num: the number of experts, provid choice 2.

    '''

    def __init__(self, expert, expert_dim, expert_num, ablation='dmoe', learnable_vec=0.):
        super().__init__()
        self.ablation = ablation
        self.expert_num = expert_num
        self.expert_dim = expert_dim

        # 添加批量归一化层
        self.norm = nn.BatchNorm2d(expert_dim)

        for i in range(self.expert_num):
            expert_i = deepcopy(expert)
            setattr(self, f"expert{i + 1}", expert_i)

        # 定义投影层
        self.proj = nn.Conv2d(expert_dim, expert_dim, kernel_size=1) if ablation != 'moe' else nn.Identity()

        # 定义激活函数
        self.activation = nn.ReLU(inplace=True)  # 你可以根据需要选择其他激活函数

        # α参数初始化策略
        if learnable_vec < 0:  # 可学习模式
            self.alpha = nn.Parameter(torch.ones((expert_dim, 1, 1)), requires_grad=True)
        else:  # 固定参数模式
            self.alpha = nn.Parameter(torch.tensor(learnable_vec, dtype=torch.float32), requires_grad=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W]
        if self.ablation == 'moe':
            atten_NE_not_i = torch.zeros_like(x)
            for i in range(self.expert_num):
                expert_i = getattr(self, f"expert{i + 1}")
                atten_NE_not_i += expert_i(x)
            out = self.proj(atten_NE_not_i)
        else:
            beta = 1 / (self.expert_num - 1)
            atten_NE_not_i = torch.zeros_like(x)
            for i in range(self.expert_num - 1):
                expert_i = getattr(self, f"expert{i + 1}")
                atten_NE_not_i += expert_i(x)
            expert_end = getattr(self, f"expert{self.expert_num}")
            out = atten_NE_not_i * beta - expert_end(x) * self.alpha

        # 在 proj 层之前添加批量归一化
        out = self.norm(out)
        out = self.proj(out)
        out = self.activation(out)  # 应用激活函数

        return out + x


# class GatingNetwork(nn.Module):
#     def __init__(self, expert_dim, num_experts):
#         super(GatingNetwork, self).__init__()
#         self.fc = nn.Linear(expert_dim, num_experts)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         x_flat = x.mean([-2, -1])  # [B, C, H, W] -> [B, C]
#         logits = self.fc(x_flat)  # [B, num_experts]
#         gate_weights = F.softmax(logits, dim=-1)  # [B, num_experts]
#
#         # Split gate_weights into two tensors
#         gate_weights_add = gate_weights[:, :-1].unsqueeze(-1).unsqueeze(-1)  # [B, num_experts-1]
#         gate_weights_sub = gate_weights[:, -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
#         return [gate_weights_add, gate_weights_sub]

class DifferenceMoEWithRouter(nn.Module):
    '''
    Difference MoE Module
    param
    ablation:
        moe: e1+e2
        dmoe: e1-alpha*e2
    expert_num: the number of experts, provid choice 2.

    '''

    def __init__(self, expert, expert_dim, expert_num, ablation='dmoe', learnable_vec=0.):
        super().__init__()
        self.ablation = ablation
        self.expert_num = expert_num
        self.norm = nn.Identity()  # nn.BatchNorm2d(expert_dim)

        for i in range(self.expert_num):
            expert_i = expert
            setattr(self, f"expert{i + 1}", expert_i)

        self.proj = nn.Conv2d(expert_dim, expert_dim, kernel_size=1, bias=True) if ablation != 'moe' else nn.Identity()

        # self.gate = GatingNetwork(expert_dim, expert_num)

    def forward(self, x):
        # [B, C, H, W]
        if self.ablation == 'moe':
            atten_NE_not_i = torch.zeros_like(x)
            for i in range(self.expert_num):
                expert_i = getattr(self, f"expert{i + 1}")
                atten_NE_not_i += expert_i(x)
            out = self.proj(atten_NE_not_i)
        else:
            gate_weights_add, gate_weights_sub = self.gate(x)
            expert_outputs_add = torch.stack(
                [getattr(self, f"expert{i + 1}")(x) for i in range(self.expert_num - 1)],
                dim=1
            )
            weighted_sum = torch.sum(gate_weights_add * expert_outputs_add, dim=1)

            expert_outputs_sub = getattr(self, f"expert{self.expert_num}")(x)
            out = weighted_sum + expert_outputs_sub * gate_weights_sub
            out = self.proj(out)

        return self.norm(out + x)


class SimplyDifferenceMoE(nn.Module):
    '''
    Difference MoE Module
    you can simply use this module to replace the original FFN or Upscale module in the Decoder or Encoder.
    param 
    expert: the module of each expert, you can use any module here. but we suggest to use SimplyExpert or any modules that have bottleneck architecture.
    expert_num: the number of experts, provid choice 2.
    '''
    def __init__(self, expert, in_dim, expert_dim,  expert_num, learnable_vec=0., proj_use_norm=True):
        super().__init__()
        self.expert_num = expert_num
        
        self.proj_input = nn.Sequential(
            nn.Conv2d(in_dim, expert_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(expert_dim),
            nn.ReLU(inplace=True)
            ) if proj_use_norm else nn.Conv2d(in_dim, expert_dim, kernel_size=3, padding=1)

        for i in range(self.expert_num):
                expert_i = expert
                setattr(self, f"expert{i + 1}", expert_i)
        
        self.proj = nn.Conv2d(expert_dim, expert_dim, kernel_size=1) 
        if learnable_vec < 0:
            self.alpha = nn.Parameter(torch.ones((expert_dim, 1, 1)), requires_grad=True)
        else:
            self.alpha = learnable_vec
    def forward(self, x):
        x = self.proj_input(x)

        beta = 1 / (self.expert_num-1)
        atten_NE_not_i = torch.zeros_like(x)
        for i in range(self.expert_num-1):
            expert_i = getattr(self, f"expert{i + 1}")
            atten_NE_not_i += expert_i(x)
            
        expert_end = getattr(self, f"expert{self.expert_num}")
        out = atten_NE_not_i * beta - expert_end(x) * self.alpha
        out = self.proj(out)

        return out+x



class SimplyExpert(nn.Module):
    
    def __init__(self, in_channels, out_channels, bottle_factor=2, drop_path_rate=0., use_moe=True ):
        super().__init__()

        self.e = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//bottle_factor, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//bottle_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//bottle_factor, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Dropout(drop_path_rate)
        )  if use_moe else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )
 
    def forward(self, x):
        return self.e(x)


#----------------------------------------------------------------------test---------------------------------------------
class CooperativeGamesMoE(nn.Module):
    '''
    Cooperative Games Module
    param ablation:
        moe: e1+e2
        dmoe: e3(e1-alpha*e2)
        ab1: 2-person  e1-alpha*e2
        ab2: 3-person  e1+e2-alpha*e3
        ab3: 3-person  alpha*(e1+e2-e3)
        ab4: 3-moe     e1+e2+e3
    '''
    def __init__(self, expert, expert_dim, ablation='dmoe', learnable_vec=0., layerscale_value=1e-4):
        super().__init__()
        self.ablation = ablation
        self.expert1 = expert
        self.expert2 = expert
        self.expert3 = expert if ablation == 'moe' else nn.Identity()
        if learnable_vec < 0:
            self.lamda = nn.Parameter(torch.ones((expert_dim, 1, 1)), requires_grad=True)
        else:
            self.lamda = learnable_vec
    def forward(self, x):
        # [B, C, H, W]
        attn_1 = self.expert1(x)
        attn_2 = self.expert2(x)
        if self.ablation == 'moe':
            out = attn_1 + attn_2
        elif self.ablation == 'ab1':
            out = attn_1 - attn_2 * self.lamda
        elif self.ablation == 'ab2':
            attn_3 = self.expert3(x)
            out = attn_1 + attn_2 - (attn_3 * self.lamda)
        elif self.ablation == 'ab3':
            attn_3 = self.expert3(x)
            out = (attn_1 + attn_2 - attn_3) * self.lamda
        elif self.ablation == 'ab4':
            attn_3 = self.expert3(x)
            out = (attn_1 + attn_2 + attn_3)
        else:
            cooperative_value = attn_1 - (attn_2 * self.lamda)
            out = self.expert3(cooperative_value)
        return out+x


class SimplyCooperativeGamesMoE(nn.Module):
    def __init__(self, expert, in_dim, expert_dim, learnable_vec=0.5, proj_use_norm=False):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, expert_dim, kernel_size=3, padding=1) if not proj_use_norm else nn.Sequential(
            nn.Conv2d(in_dim, expert_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(expert_dim),
            nn.ReLU(inplace=True)
            )

        self.expert1 = expert
        self.expert2 = expert
        #self.expert3 = expert
        if learnable_vec < 0:
            self.lamda = nn.Parameter(torch.ones((expert_dim, 1, 1)), requires_grad=True)
        else:
            self.lamda = learnable_vec
    def forward(self, x):
        x = self.proj(x)
        # [B, C, H, W]
        attn_1 = self.expert1(x)
        attn_2 = self.expert2(x)
        out = attn_1 - (attn_2 * self.lamda)
        # cooperative_value = attn_1 - (attn_2 * self.lamda)
        # out = self.expert3(cooperative_value)
        return out+x
