import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule



class CrossAttnMem(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels 
        self.q_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.q_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)

        self.out_u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.out_l2u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        self.pseudo_label = None
        self.SMem_size = embedding_channels
        self.register_buffer("queue_ptr", torch.zeros(self.num_class, dtype=torch.long))
        self.register_buffer("kv_queue", torch.randn(self.num_class, self.SMem_size, self.patch_num))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, emb_u_, channel_cls):
        """
        Semantic Memory (S-Mem): first-in-first-out queue
        """
        label = torch.unique(channel_cls).detach().cpu().numpy()
        label = np.delete(label, np.where(label == 255))
        for cls_i in label:
            channel_cls_i = torch.where(channel_cls == int(cls_i), 1, 0)
            index = torch.squeeze(torch.nonzero(channel_cls_i), dim=-1)
            emb_u_i = emb_u_[:, index].transpose(-2, -1)
            channel_num = emb_u_i.size(0)
            # obtain the pointer
            ptr_i = int(self.queue_ptr[cls_i])
            # replace the keys at ptr (dequeue and enqueue)
            if ptr_i + channel_num > self.SMem_size: # queue full
                self.kv_queue[cls_i, ptr_i:, :] = emb_u_i[:self.SMem_size - ptr_i, :]
                self.kv_queue[cls_i, :ptr_i + channel_num - self.SMem_size, :] = emb_u_i[self.SMem_size - ptr_i:, :]
                new_ptr = ptr_i + channel_num - self.SMem_size
            else:  # queue not full
                self.kv_queue[cls_i, ptr_i:ptr_i + channel_num, :] = emb_u_i
                new_ptr = (ptr_i + channel_num) % self.SMem_size  # move pointer
            self.queue_ptr[cls_i] = new_ptr

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb, pseudo_label, pseudo_prob_map, using_SMem):

        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_u.size()


        if self.training:
            with torch.no_grad():
                # Channel-wise Semantic Grouping
                pseudo_prob_map = F.interpolate(pseudo_prob_map, size=(int(math.sqrt(N)), int(math.sqrt(N))),
                                                mode="bilinear", align_corners=False)

                pseudo_prob_map = torch.flatten(pseudo_prob_map, start_dim=2, end_dim=-1)
                cls_sim_matrix = torch.matmul(F.normalize(pseudo_prob_map, dim=-1), F.normalize(emb_u, dim=1))
                channel_cls = torch.argmax(cls_sim_matrix, dim=1)

                # update the Semantic Memory (S-Mem)
                for b in range(emb_u.size(0)):
                    self._dequeue_and_enqueue(emb_u[b, :, :], channel_cls[b, :])

        q_u = self.q_u(emb_u)
        k_u = self.k_u(emb_u)
        v_u = self.v_u(emb_u)


        # convert to multi-head representation
        mh_q_u = self.multi_head_rep(q_u).transpose(-1, -2)
        mh_k_u = self.multi_head_rep(k_u)
        mh_v_u = self.multi_head_rep(v_u).transpose(-1, -2)

        self_attn_u = torch.matmul(mh_q_u, mh_k_u)

        self_attn_u = self.attn_dropout(self.softmax(self.psi(self_attn_u)))
        self_attn_u = torch.matmul(self_attn_u, mh_v_u)

        self_attn_u = self_attn_u.permute(0, 3, 2, 1).contiguous()
        new_shape_u = self_attn_u.size()[:-2] + (self.KV_size,)
        self_attn_u = self_attn_u.view(*new_shape_u)

        out_u = self.out_u(self_attn_u)
        out_u = self.proj_dropout(out_u)

        # ==========================================================

        q_l2u = self.q_l2u(emb_l)
        if using_SMem and self.training:
            features_mem = self.kv_queue.clone()
            features_mem_re = features_mem.transpose(-1, -2)
            k_l2u = self.k_l2u(features_mem_re)
            v_l2u = self.v_l2u(features_mem_re)
        else:
            k_l2u = self.k_l2u(emb_u)
            v_l2u = self.v_l2u(emb_u)

        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)


        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        # if using_SMem:
        #     cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u) / self.temperature))
        # else:
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u, v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out_l2u(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        out = torch.cat([out_l2u, out_u], dim=0)
        return out



class AllSpark(nn.Module):
    def __init__(self, num_heads, embedding_channels, channel_num, channel_num_out,
                 attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = CrossAttnMem(num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num)
        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en, pseudo_label, pseudo_prob_map, using_SMem):
        if not self.training:
            en = torch.cat((en, en))

        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)

        emb = self.attn_norm(en)
        emb = self.attn(emb, pseudo_label, pseudo_prob_map, using_SMem)
        emb = emb + en

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        if not self.training:
            out = torch.split(out, out.size(0) // 2, dim=0)[0]

        return out




class SemiDecoder(nn.Module):
    def __init__(self, num_heads, num_class, in_planes, image_size, warmup_epoch, embedding_dim):
        super(SemiDecoder, self).__init__()

        self.pseudo_label = None
        self.pseudo_prob_map = None
        self.using_SMem = False
        self.warmup_epoch = warmup_epoch
        self.allspark = AllSpark(num_heads=num_heads,
                                 embedding_channels=in_planes[3],
                                 channel_num=in_planes[3],
                                 channel_num_out=in_planes[3],
                                 attention_dropout_rate=0.1,
                                 patch_num=(image_size//32+1)**2,
                                 num_class=num_class)

        self.decoder = SegFormerHead(embedding_dim, in_planes, num_class)


    def set_pseudo_label(self, pseudo_label):
        self.pseudo_label = pseudo_label

    def set_pseudo_prob_map(self, pseudo_prob_map):
        self.pseudo_prob_map = pseudo_prob_map

    def set_SMem_status(self, epoch, isVal=False):
        if epoch >= self.warmup_epoch and not isVal:
            self.using_SMem = True
        else:
            self.using_SMem = False

    def forward(self, feats, h, w):
        e1, e2, e3, e4 = feats

        e4_allspark = self.allspark(e4, self.pseudo_label, self.pseudo_prob_map, self.using_SMem)

        out = self.decoder(e1, e2, e3, e4_allspark)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, embedding_dim, in_channels, num_class):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(0.1)

        self.linear_pred = nn.Conv2d(embedding_dim, num_class, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
