import pytorch_lightning as pl  # 导入PyTorch Lightning库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
from torch_geometric.data import Batch  # 从torch_geometric.data模块导入Batch类
from torch_geometric.utils import unbatch  # 从torch_geometric.utils模块导入unbatch函数
import math  # 导入数学库
import pandas as pd  # 导入Pandas库
 
from losses import HuberTrajLoss  # 从losses模块导入HuberTrajLoss类
from metrics import minJointADE  # 从metrics模块导入minJointADE函数
from metrics import minJointFDE  # 从metrics模块导入minJointFDE函数
from modules import Backbone  # 从modules模块导入Backbone类
from modules import MapEncoder  # 从modules模块导入MapEncoder类
 
from utils import generate_target  # 从utils模块导入generate_target函数
from utils import generate_predict_mask  # 从utils模块导入generate_predict_mask函数
from utils import compute_angles_lengths_2D  # 从utils模块导入compute_angles_lengths_2D函数
#torch.set_float32_matmul_precision('high')

# 定义HPNet类，继承自pl.LightningModule
class HPNet(pl.LightningModule):

    # 初始化方法，设置模型的超参数
    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 **kwargs) -> None:
        super(HPNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max

        # 初始化Backbone对象，配置模型参数
        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            duration=duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout
        )
        # 初始化MapEncoder实例，用于映射编码
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        # 定义一个类的构造函数
        # 初始化损失函数和其他属性
        self.reg_loss_traj = HuberTrajLoss()
        
        # 初始化最小关节ADE和FDE
        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()
        
        # 定义数据列名称
        self._columns = ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'interesting_agent',
                         'x1', 'y1', 'psi_rad1',
                         'x2', 'y2', 'psi_rad2',
                         'x3', 'y3', 'psi_rad3',
                         'x4', 'y4', 'psi_rad4',
                         'x5', 'y5', 'psi_rad5',
                         'x6', 'y6', 'psi_rad6']
        
        # 初始化测试输出字典
        self.test_output = dict()

    def forward(self, data: Batch):
        """执行前向传播，计算模型输出。
         
        参数:
            data (Batch): 输入的数据批次
 
        返回:
            pred: 模型的预测输出
        """
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def training_step(self,data,batch_idx):
        """执行一次训练步骤，计算预测轨迹并计算损失。
         
        参数:
            data: 输入的数据，包含代理的状态和可见性掩码等信息。
            batch_idx: 当前批次的索引。
         
        返回:
            loss: 计算得到的损失值。
        """
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F],[(N1,...,Nb),H,K]
         
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
 
        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]
 
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
         
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]
 
        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
         
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]
 
        reg_loss_traj_propose = self.reg_loss_traj(traj_pro[targ_mask], targ_traj[targ_mask]) 
        reg_loss_traj_refine = self.reg_loss_traj(traj_ref[targ_mask], targ_traj[targ_mask])  
        loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('train_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
 
        return loss

    # 验证步骤，在每个批次上执行验证评估
    def validation_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]
         
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
 
        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
         
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]
 
        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
         
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]
         
        reg_loss_traj_propose = self.reg_loss_traj(traj_pro[targ_mask], targ_traj[targ_mask]) 
        reg_loss_traj_refine = self.reg_loss_traj(traj_ref[targ_mask], targ_traj[targ_mask])   
        loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('val_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
 
        visible_mask = data['agent']['visible_mask'][agent_mask]                      #[(n1,...nb),H+F]
        visible_num = visible_mask.sum(dim=-1)                                        #[(n1,...nb)]
        scored_mask = visible_num == self.num_historical_steps + self.num_future_steps
        scored_predict_traj = unbatch(traj_output[scored_mask,-1], agent_batch[scored_mask])                   #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_target_traj = unbatch(target_traj[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F,2),...,(nb,F,2)]
        scored_target_mask = unbatch(target_mask[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F),...,(nb,F)]
 
        self.min_joint_ade.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.min_joint_fde.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_minJointADE', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_minJointFDE', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def test_step(self,data,batch_idx):
        """
        执行测试步骤，处理输入数据并生成预测轨迹和角度输出
        参数:
        data: 输入数据，包括代理信息和场景信息
        batch_idx: 批次索引
        返回:
        无
        """
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F]
 
        agent_mask = data['agent']['category'] == 1
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        num_agent_pre_batch = torch.bincount(agent_batch)
 
        scenario_name = data['scenario_name']   #[b]
        case_id = data['case_id']               #[b]
        agent_id = data['agent']['id'][agent_mask]
        agent_interset = data['agent']['interest'][agent_mask]
        traj_output = traj_output[agent_mask, -1]
        tep = torch.cat([data['agent']['position'][agent_mask, -1:].unsqueeze(1).repeat_interleave(self.num_modes,1), traj_output], dim=-2) #[(n1+...+nb),K,F+1,2]
        _, yaw_output = compute_angles_lengths_2D(tep[:,:,1:] - tep[:,:,:-1])   #[(n1+...+nb),K,F]
 
        scored_agent_id = unbatch(agent_id, agent_batch)                        #[n1,...nb]
        scored_agent_interset = unbatch(agent_interset, agent_batch)            #[n1,...nb]
        scored_predict_traj = unbatch(traj_output, agent_batch)           #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_predict_yaw = unbatch(yaw_output, agent_batch)             #[(n1,K,F),...,(nb,K,F)]
         
        case_id = case_id.cpu().numpy()
        scored_agent_id = [agent_id.cpu().numpy() for agent_id in scored_agent_id]
        scored_agent_interset = [agent_interset.cpu().numpy() for agent_interset in scored_agent_interset]
        scored_predict_traj = [predict_traj.cpu().numpy() for predict_traj in scored_predict_traj]
        scored_predict_yaw = [predict_yaw.cpu().numpy() for predict_yaw in scored_predict_yaw]
         
        scored_frame_id = list(range(30))
        scored_frame_id = [id + 11 for id in scored_frame_id]
        scored_timestamp_ms = [frame_id * 100 for frame_id in scored_frame_id]

        # 遍历批次中的每一个样本
        for i in range(batch_size):
            rows = []  # 存储当前样本的所有行数据
            # 遍历每个代理的预处理批次
            for j in range(num_agent_pre_batch[i]):
                # 遍历未来步数
                for k in range(self.num_future_steps):
                    # 构建一行数据
                    row = [case_id[i], scored_agent_id[i][j], scored_frame_id[k], scored_timestamp_ms[k], scored_agent_interset[i][j],
                        scored_predict_traj[i][j,0,k,0], scored_predict_traj[i][j,0,k,1], scored_predict_yaw[i][j,0,k],
                        scored_predict_traj[i][j,1,k,0], scored_predict_traj[i][j,1,k,1], scored_predict_yaw[i][j,1,k],
                        scored_predict_traj[i][j,2,k,0], scored_predict_traj[i][j,2,k,1], scored_predict_yaw[i][j,2,k],
                        scored_predict_traj[i][j,3,k,0], scored_predict_traj[i][j,3,k,1], scored_predict_yaw[i][j,3,k],
                        scored_predict_traj[i][j,4,k,0], scored_predict_traj[i][j,4,k,1], scored_predict_yaw[i][j,4,k],
                        scored_predict_traj[i][j,5,k,0], scored_predict_traj[i][j,5,k,1], scored_predict_yaw[i][j,5,k]]
                    rows.append(row)  # 将构建的行数据添加到 rows 中

            # 根据场景名称更新测试输出
            if scenario_name[i] in self.test_output:
                self.test_output[scenario_name[i]] = self.test_output[scenario_name[i]] + rows
            else:
                self.test_output[scenario_name[i]] = rows

    # 该方法在测试结束时被调用，用于处理测试输出并保存为CSV文件
    def on_test_end(self):
        for key, value in self.test_output.items():
            df = pd.DataFrame(value, columns=self._columns)
            df['track_to_predict'] = 1
            df.to_csv('./test_output/' + key + '_sub.csv', index=False)

    # 配置优化器并设定参数的权重衰减策略
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
 
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
 
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
         
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        # 定义一个用于热身和余弦退火调度的函数
        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))
 
        # 定义学习率调度器的配置字典
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    # 添加模型特定参数的静态方法
    @staticmethod
    def add_model_specific_args(parent_parser):
        # 创建一个参数组用于HPNet
        parser = parent_parser.add_argument_group('HPNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=10)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--duration', type=int, default=10)
        parser.add_argument('--a2a_radius', type=float, default=80)
        parser.add_argument('--l2a_radius', type=float, default=80)
        parser.add_argument('--num_visible_steps', type=int, default=3)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=3)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
