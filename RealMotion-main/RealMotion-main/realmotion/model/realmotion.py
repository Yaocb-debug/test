from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.mtr_decoder import TransformerDecoder
from .layers.transformer_blocks import Block, InteractionModule


class RealMotion_I(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps=60,
        use_transformer_decoder=False,
        num_decoder_layers=6,
    ) -> None:
        super().__init__()
        self.use_transformer_decoder = use_transformer_decoder
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.decoder = MultimodalDecoder(embed_dim, future_steps, return_prob=not use_transformer_decoder)
        if use_transformer_decoder:
            self.trans_decoder = TransformerDecoder(embed_dim, future_steps, num_decoder_layers=num_decoder_layers)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {
            k[len('net.') :]: v for k, v in ckpt.items() if k.startswith('net.')
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        hist_valid_mask = data['x_valid_mask']          
        hist_key_valid_mask = data['x_key_valid_mask']
        hist_feat = torch.cat(
            [
                data['x_positions_diff'],
                data['x_velocity_diff'][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[hist_feat_key_valid].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_valid_mask = data['lane_valid_mask']
        lane_normalized = data['lane_positions'] - data['lane_centers'].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data['x_centers'], data['lane_centers']], dim=1)
        angles = torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data['x_attr'][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [data['x_key_valid_mask'], data['lane_key_valid_mask']], dim=1
        )
        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),
                                 lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()

        x_encoder = x_encoder + pos_embed
        if isinstance(self, RealMotion):
            # read memory for stream process
            if 'memory_dict' in data and data['memory_dict'] is not None:
                rel_pos = data['origin'] - data['memory_dict']['origin']
                rel_ang = (data['theta'] - data['memory_dict']['theta'] + torch.pi) % (2 * torch.pi) - torch.pi
                rel_ts = data['timestamp'] - data['memory_dict']['timestamp']
                memory_pose = torch.cat([
                    rel_ts.unsqueeze(-1), rel_ang.unsqueeze(-1), rel_pos
                ], dim=-1).float().to(x_encoder.device)
                memory_x_encoder = data['memory_dict']['x_encoder']
                memory_valid_mask = data['memory_dict']['x_mask']
                memory_type_mask = data['memory_dict']['x_type_mask']
            else:
                memory_pose = x_encoder.new_zeros(x_encoder.size(0), self.pose_dim)
                memory_x_encoder = x_encoder
                memory_valid_mask = key_valid_mask
                memory_type_mask = x_type_mask
            cur_pose = torch.zeros_like(memory_pose)

            # scene interaction
            new_x_encoder = x_encoder
            C = x_encoder.size(-1)
            # new_x_encoder = self.scene_interact(new_x_encoder, memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
            new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
            new_lane_feat = self.scene_interact(new_x_encoder[~x_type_mask].reshape(B, -1, C), memory_x_encoder[~memory_type_mask].reshape(B, -1, C), cur_pose, memory_pose, key_padding_mask=~memory_valid_mask[~memory_type_mask].reshape(B, -1))
            new_x_encoder = torch.cat([new_actor_feat, new_lane_feat], dim=1)
            x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + x_encoder * ~key_valid_mask.unsqueeze(-1)

        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi, x_mode = self.decoder(x_agent)
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)
        
        cos, sin = data['theta'].cos(), data['theta'].sin()
        rot_mat = data['theta'].new_zeros(B, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos


        if isinstance(self, RealMotion):
            # traj interaction
            if 'memory_dict' in data and data['memory_dict'] is not None:
                memory_y_hat = data['memory_dict']['glo_y_hat']
                memory_x_mode = data['memory_dict']['x_mode']
                ori_idx = ((data['timestamp'] - data['memory_dict']['timestamp']) / 0.1).long() - 1
                memory_traj_ori = torch.gather(memory_y_hat, 2, ori_idx.reshape(
                    B, 1, -1, 1).repeat(1, memory_y_hat.size(1), 1, memory_y_hat.size(-1)))
                memory_y_hat = torch.bmm((memory_y_hat - memory_traj_ori).reshape(B, -1, 2), rot_mat
                                        ).reshape(B, memory_y_hat.size(1), -1, 2)
                traj_embed = self.traj_embed(y_hat.detach().reshape(B, y_hat.size(1), -1))
                memory_traj_embed = self.traj_embed(memory_y_hat.reshape(B, memory_y_hat.size(1), -1))
                x_mode = self.traj_interact(x_mode, memory_x_mode, cur_pose, memory_pose,
                                                    cur_pos_embed=traj_embed,
                                                    memory_pos_embed=memory_traj_embed)
                y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)
                y_hat = y_hat + y_hat_diff

        ret_dict = {
            'y_hat': y_hat,
            'pi': pi,
            'y_hat_others': y_hat_others,
        }

        glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(rot_mat))
        glo_y_hat = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)

        if isinstance(self, RealMotion):
            memory_dict = {
                'x_encoder': x_encoder,
                'x_mode': x_mode,
                'glo_y_hat': glo_y_hat,
                'x_mask': key_valid_mask,
                'x_type_mask': x_type_mask,
                'origin': data['origin'],
                'theta': data['theta'],
                'timestamp': data['timestamp'],
            }
            ret_dict['memory_dict'] = memory_dict

        return ret_dict


class RealMotion(RealMotion_I):
    def __init__(self, 
                 max_memory_size=5,  # 最大记忆库大小
                 adapt_memory_size=True,  # 是否启用自适应记忆大小
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_memory_size = max_memory_size  # 最大智能体记忆库大小
        self.adapt_memory_size = adapt_memory_size  # 是否启用自适应记忆大小
        self.embed_dim = kwargs['embed_dim']
        self.pose_dim = 4
        
        # 初始化其他组件...
        self.scene_interact = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.traj_interact = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )
         
    
        self.stream_loc = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, kwargs['embed_dim']),
            nn.ReLU(),
            nn.Linear(kwargs['embed_dim'], kwargs['future_steps'] * 2),
        )
        self.traj_embed = nn.Sequential(
            nn.Linear(kwargs['future_steps'] * 2, kwargs['embed_dim']),
            nn.GELU(),
            nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
        )

    def update_memory(self, data, x_encoder, key_valid_mask, x_type_mask):
        """
        更新记忆库，限制共享记忆库的大小，并动态选择重要智能体。
        """
        memory_dict = {
            'x_encoder': x_encoder,
            'x_mode': data['x_mode'],  # 当前智能体的模式
            'glo_y_hat': data['glo_y_hat'],  # 当前智能体的预测轨迹
            'x_mask': key_valid_mask,
            'x_type_mask': x_type_mask,
            'origin': data['origin'],
            'theta': data['theta'],
            'timestamp': data['timestamp'],
        }

        # 计算每个智能体与当前智能体的相似度（例如，基于位置、速度等）
        if self.adapt_memory_size:
            memory_dict = self.adaptive_memory_size(data, memory_dict)
        
        # 限制记忆库的大小，使用FIFO策略
        if len(memory_dict['x_encoder']) > self.max_memory_size:
            memory_dict = self.apply_fifo(memory_dict)
        
        return memory_dict

    def apply_fifo(self, memory_dict):
        """
        应用FIFO策略，限制记忆库的大小，保留最新的max_memory_size个智能体。
        """
        memory_dict['x_encoder'] = memory_dict['x_encoder'][-self.max_memory_size:]  # 保留最近的max_memory_size个智能体
        memory_dict['x_mode'] = memory_dict['x_mode'][-self.max_memory_size:]
        memory_dict['glo_y_hat'] = memory_dict['glo_y_hat'][-self.max_memory_size:]
        memory_dict['x_mask'] = memory_dict['x_mask'][-self.max_memory_size:]
        memory_dict['x_type_mask'] = memory_dict['x_type_mask'][-self.max_memory_size:]
        memory_dict['origin'] = memory_dict['origin'][-self.max_memory_size:]
        memory_dict['theta'] = memory_dict['theta'][-self.max_memory_size:]
        memory_dict['timestamp'] = memory_dict['timestamp'][-self.max_memory_size:]
        return memory_dict

    def adaptive_memory_size(self, data, memory_dict):
        """
        基于场景复杂度或智能体的密度来动态调整记忆库的大小。
        """
        # 这里假设场景的复杂度与当前智能体的数量成正比
        num_agents = data['num_agents']  # 当前场景中的智能体数量
    
        # 根据场景复杂度动态调整最大记忆库大小
        # 如果智能体数量较多，增加记忆库大小；反之，减少记忆库大小
        if num_agents > 5:  # 如果智能体数量较多
          self.max_memory_size = min(self.max_memory_size + 2, 5)  # 增加记忆库大小，但不超过 max_memory_size

        elif num_agents < 2:  # 如果智能体数量较少
          self.max_memory_size = max(self.max_memory_size - 1, 2)  # 减少记忆库大小，但保持至少有 2 个记忆库容量
        
        return memory_dict

    def retrieve_memory(self, data, memory_dict):
        """
        从记忆库中检索当前智能体的共享记忆。
        """
        memory_x_encoder = memory_dict['x_encoder']
        memory_valid_mask = memory_dict['x_mask']
        memory_type_mask = memory_dict['x_type_mask']
        return memory_x_encoder, memory_valid_mask, memory_type_mask

    def forward(self, data):
        # 更新记忆库
        if 'x_encoder' in data:
            # 如果 x_encoder 已经在 data 中，则更新记忆库
            memory_dict = self.update_memory(data, data['x_encoder'], data['key_valid_mask'], data['x_type_mask'])
        else:
            # 如果 x_encoder 不在 data 中，暂时跳过更新
            memory_dict = None  # 或者保留原有的 memory_dict

        # 进行进一步的处理，使用memory_dict中的共享记忆...
        
        if 'y_hat' in data and 'pi' in data and 'y_hat_others' in data:
            ret_dict = {
                'y_hat': data['y_hat'],
                'pi': data['pi'],
                'y_hat_others': data['y_hat_others'],
                    }
        else:
            # 可能需要先进行其他处理，或者给默认值
            ret_dict = {}


        # 将更新后的共享记忆库返回
        ret_dict['memory_dict'] = memory_dict

        return ret_dict

        
