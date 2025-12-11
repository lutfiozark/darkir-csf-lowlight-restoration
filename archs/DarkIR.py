import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from arch_model import EBlock, DBlock
    from arch_util import CustomSequential
except:
    from archs.arch_model import EBlock, DBlock
    from .arch_util import CustomSequential


class CSFBlock(nn.Module):
    """
    Cross-Scale Fusion block. Learns a per-channel gate to balance encoder and decoder features.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, enc_feat, dec_feat):
        summary = self.fuse(torch.cat([enc_feat, dec_feat], dim=1))
        gate = self.gate(summary)
        return gate * enc_feat + (1 - gate) * dec_feat

class DarkIR(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations = [1, 4, 9], 
                 extra_depth_wise = True,
                 use_csf: bool = False,
                 csf_reduction: int = 4):
        super(DarkIR, self).__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.use_csf = use_csf
        self.csf_blocks = nn.ModuleList() if use_csf else None
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if self.use_csf:
                self.csf_blocks.append(CSFBlock(chan, reduction=csf_reduction))
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)        
        
        # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
        self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel, 
                                kernel_size = 3, stride=1, padding=1)
        
    def forward(self, input, side_loss = False, use_adapter = None):

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(input)
        
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        if self.use_csf:
            for decoder, up, skip, csf in zip(self.decoders, self.ups, skips[::-1], self.csf_blocks):
                x = up(x)
                x = csf(skip, x)
                x = decoder(x)
        else:
            for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
                x = up(x)
                x = x + skip
                x = decoder(x)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W] # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x      


class DarkIRCSF(DarkIR):
    """
    Cross-Scale Fusion variant of DarkIR. Keeps encoder/decoder blocks but replaces skip merges with CSF.
    """
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations = [1, 4, 9], 
                 extra_depth_wise = True,
                 csf_reduction: int = 4):
        super().__init__(img_channel=img_channel, 
                         width=width, 
                         middle_blk_num_enc=middle_blk_num_enc,
                         middle_blk_num_dec=middle_blk_num_dec, 
                         enc_blk_nums=enc_blk_nums, 
                         dec_blk_nums=dec_blk_nums,  
                         dilations=dilations, 
                         extra_depth_wise=extra_depth_wise,
                         use_csf=True,
                         csf_reduction=csf_reduction)

if __name__ == '__main__':
    
    img_channel = 3
    width = 32
    
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net = DarkIR(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise)
    
    new_state_dict = net.state_dict()

    inp_shape = (3, 256, 256)

    net.load_state_dict(new_state_dict)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    
    weights = net.state_dict()
    adapter_weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    
    
