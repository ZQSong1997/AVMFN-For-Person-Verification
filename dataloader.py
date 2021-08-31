import numpy as np
from torch.utils import data



# cat_embeds.append(Cat_emb)	# cat_paths.append(Cat_path)	# cat_labels.append(Class_Label)
class Dev_Embd_Dataset(data.Dataset):
	def __init__(self, cat_embeds, cat_labels):
		self.cat_embeds = cat_embeds
		self.cat_labels = cat_labels

	def __len__(self):              	# ``__len__``, 数据集的大小，cat_embeds训练集有10W个，测试集1W个
		return len(self.cat_embeds)

	def __getitem__(self, index):      # ``__getitem__``, 支持范围从 0 到 len(self) 的整数索引。
		X = self.cat_embeds[index]
		X = X.astype(np.float32)
		y = self.cat_labels[index]     # 标签 0-1210
		return X, y

class Val_Embd_Dataset(data.Dataset):
	def __init__(self, cat_embeds):
		self.cat_embeds = cat_embeds
		
	def __len__(self):              	# ``__len__``, 数据集的大小，cat_embeds训练集有10W个，测试集1W个
		return len(self.cat_embeds)

	def __getitem__(self, index):      # ``__getitem__``, 支持范围从 0 到 len(self) 的整数索引。
		X = self.cat_embeds[index]
		X = X.astype(np.float32)
		return X






