# 引入所需的库和模块
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from datasets import INTERACTIONDataset

from transforms import HorizontalFlip
from transforms import AgentRandomOcclusion
from transforms import LaneRandomOcclusion


# INTERACTION 数据模块，用于处理训练和验证数据集
class INTERACTIONDataModule(pl.LightningDataModule):

    # 初始化方法，设置数据模块的参数
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 flip_p: float = 0.5,
                 agent_occlusion_ratio: float = 0.0,
                 lane_occlusion_ratio: float = 0.2,
                 num_historical_steps: int = 10,
                 num_future_steps: int = 30,
                 **kwargs) -> None:
        super(INTERACTIONDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transform = Compose([HorizontalFlip(flip_p), AgentRandomOcclusion(agent_occlusion_ratio,num_historical_steps), LaneRandomOcclusion(lane_occlusion_ratio)])
        self.val_transform = LaneRandomOcclusion(0.0)
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    # 准备训练和验证数据集
    def prepare_data(self) -> None:
        INTERACTIONDataset(self.root, 'train', self.train_transform, self.num_historical_steps, self.num_future_steps)
        INTERACTIONDataset(self.root, 'val', self.val_transform, self.num_historical_steps, self.num_future_steps)

    # 设置训练和验证数据集
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = INTERACTIONDataset(self.root, 'train', self.train_transform, self.num_historical_steps, self.num_future_steps)
        self.val_dataset = INTERACTIONDataset(self.root, 'val', self.val_transform, self.num_historical_steps, self.num_future_steps)

    # 获取训练数据加载器
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    # 获取验证数据加载器
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)