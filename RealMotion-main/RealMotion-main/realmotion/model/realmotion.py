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
        closest_agents_num=8,
    ) -> None:
        super().__init__()
        self.use_transformer_decoder = use_transformer_decoder
        self.closest_agents_num=closest_agents_num # 初始化 closest_agents_num
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

    def select_closest_agents(self, x_encoder_agents: torch.Tensor, hist_feat: torch.Tensor, hist_valid_mask: torch.Tensor) -> torch.Tensor:

        """
            参数:
                x_encoder_agents (Tensor): 输入的智能体特征矩阵，形状为 (B, num_agents, feature_dim)。
                hist_feat (Tensor): 历史轨迹特征，形状为 (B*num_agents, L, D)。
                hist_valid_mask (Tensor): 历史轨迹有效性掩码，形状为 (B, num_agents, L)。
        """
        # 获取 B, num_agents, feature_dim 的维度
        B, num_agents, feature_dim = x_encoder_agents.shape
        center_agent = x_encoder_agents[:, 0, :]  # 中心智能体
        others = x_encoder_agents[:, 1:, :]       # 其他智能体

        #重塑hist_feat为4维
        L, D = hist_feat.shape[1], hist_feat.shape[2]
        hist_feat_4d = hist_feat.view(B, num_agents, L, D)  # 重塑为 (B, num_agents, L, D)

        # (1) 计算欧式距离
        distances = torch.norm(others - center_agent[:, None, :], dim=-1)   # (B, num_agents-1)

        # (2) 从hist_feat提取速度方向
        # 取最后两个时间步的位置差分，计算方向向量
        pos_diff = hist_feat_4d[..., -2:, :2]  # (B, num_agents, 2, 2)
        direction = pos_diff[..., 1, :] - pos_diff[..., 0, :]  # (B, num_agents, 2)
        center_direction = direction[:, 0, :]  # (B, 2)
        others_direction = direction[:, 1:, :]  # (B, num_agents-1, 2)

        # 归一化方向向量并计算方向相似性
        center_direction = F.normalize(center_direction, dim=-1) + 1e-8
        others_direction = F.normalize(others_direction, dim=-1) + 1e-8
        velocity_sim = F.cosine_similarity(others_direction, center_direction[:, None, :], dim=-1)

        # (3) 计算历史轨迹相关性
        hist_embed = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
        hist_embed = hist_embed.view(B, num_agents, -1)
        center_hist_embed = hist_embed[:, 0, :]
        others_hist_embed = hist_embed[:, 1:, :]
        traj_sim = F.cosine_similarity(others_hist_embed, center_hist_embed[:, None, :], dim=-1)

        # (4) 综合得分
        w1, w2, w3 = 0.3, 0.5, 0.4  # 距离权重:0.3, 方向权重:0.5, 轨迹权重:0.4
        score = w1 * distances + w2 * (1 - velocity_sim.clamp(min=0)) + w3 * (1 - traj_sim.clamp(min=0))
        
        # (5) 选择得分最低的N个智能体
        _, indices = torch.topk(score, self.closest_agents_num, dim=-1, largest=False)
        closest_agents = torch.gather(
            others, dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        )
        
        return closest_agents, indices

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

            # 场景交互
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

        cos, sin = data['theta'].cos(), data['theta'].sin()
        rot_mat = data['theta'].new_zeros(B, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        # Select the closest N agents (in this case, 3 closest agents)
        x_encoder_agents = x_encoder[:, :N, :]  # 只取智能体部分
        closest_agents, closest_agents_idx = self.select_closest_agents(x_encoder_agents, hist_feat, hist_valid_mask)

        x_agent = x_encoder[:, 0]
        y_hat, pi, x_mode = self.decoder(x_agent)

        # Decode the closest N agents using the same decoder as the center agent
        x_modes_closest_agents = []  # 新增：收集最近N个智能体的x_mode
        y_hat_closest = []  # 新增轨迹容器 
        glo_y_hat_closest_list = []
        for i in range(closest_agents.size(1)):
            agent_feat = closest_agents[:, i, :]
            y_hat_agent, pi_agent, x_mode_agent = self.decoder(agent_feat)
           
            glo_y_hat_closest = torch.bmm(y_hat_agent.detach().reshape(B, -1, 2), torch.inverse(rot_mat))
            glo_y_hat_closest = glo_y_hat_closest.reshape(B, y_hat.size(1), -1, 2)

            x_modes_closest_agents.append(x_mode_agent)  # 单独收集x_mode
            y_hat_closest.append(y_hat_agent)
            glo_y_hat_closest_list.append(glo_y_hat_closest)
        x_modes_closest_agents = torch.stack(x_modes_closest_agents, dim=1)  
        y_hat_closest = torch.stack(y_hat_closest,  dim=1)
        glo_y_hat_closest_list = torch.stack(glo_y_hat_closest_list, dim=1)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

       
        # 下面的x_mode才是修改的重点，现在是只有中心智能体有自己的x_mode，其余智能体没有
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
                x_mode = self.traj_interact_center(x_mode, memory_x_mode, cur_pose, memory_pose,
                                                    cur_pos_embed=traj_embed,
                                                    memory_pos_embed=memory_traj_embed)
                

                 # 邻近智能体记忆交互 
                updated_y_hat_closest = []
                updated_x_modes = []
                for i in range(x_modes_closest_agents.size(1)): 
                    # 每个邻近智能体独立处理 
                    x_mode_agent = x_modes_closest_agents[:, i]
                    y_hat_agent = y_hat_closest[:, i]  # 新增当前轨迹

                    
                    memory_x_mode_agent = data['memory_dict']['x_mode_closest'][:, i]
                    memory_y_hat_agent = data['memory_dict']['glo_y_hat_closest_list'][:, i]

                    memory_traj_ori = torch.gather(memory_y_hat_agent, 2, ori_idx.reshape(
                    B, 1, -1, 1).repeat(1, memory_y_hat_agent.size(1), 1, memory_y_hat_agent.size(-1)))
                    memory_y_hat_agent = torch.bmm((memory_y_hat_agent - memory_traj_ori).reshape(B, -1, 2), rot_mat
                                            ).reshape(B, memory_y_hat_agent.size(1), -1, 2)
                    traj_embed = self.traj_embed(y_hat_agent.detach().reshape(B, y_hat_agent.size(1), -1))
                    memory_traj_embed = self.traj_embed(memory_y_hat_agent.reshape(B, memory_y_hat_agent.size(1), -1))
                    x_mode_agent = self.traj_interact_agent(x_mode_agent, memory_x_mode_agent, cur_pose, memory_pose,
                                                        cur_pos_embed=traj_embed,
                                                        memory_pos_embed=memory_traj_embed)
                    
                    x_mode = self.traj_interact_center_agent(x_mode, x_mode_agent, cur_pose, memory_pose,
                                                    cur_pos_embed=traj_embed,
                                                    memory_pos_embed=memory_traj_embed)

                    updated_x_modes.append(x_mode_agent)
                    updated_y_hat_closest.append(y_hat_agent)

                # 合并更新后的模式
                x_modes_closest_agents = torch.stack(updated_x_modes,  dim=1)
                glo_y_hat_closest_list = torch.stack(updated_y_hat_closest,  dim=1)



                y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)  # 只更新中心智能体
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
                'x_mode_closest': x_modes_closest_agents,# 新增邻近智能体记忆存储  
                'glo_y_hat': glo_y_hat,
                'glo_y_hat_closest_list': glo_y_hat_closest_list, # 添加邻近智能体全局轨迹存储
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
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = kwargs['embed_dim']
        self.pose_dim = 4
        # self.use_stream_encoder = use_stream_encoder
        # self.use_stream_decoder = use_stream_decoder
        
        # 初始化其他组件...
        self.scene_interact = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.traj_interact_center = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.traj_interact_agent = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.traj_interact_center_agent = InteractionModule(
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
    
    

        
