import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PreTrainDataset(Dataset):
    def __init__(self, data, mask_ratio=0.3):
        self.data = data  # 形状：[num_samples, seq_length, 2]
        self.mask_ratio = mask_ratio
        self.processed_data = self._preprocess()

    def _preprocess(self):
        """IQ→幅相转换+归一化"""
        processed = []
        for sample in tqdm(self.data, desc="幅相转换与归一化"):
            I = sample[:, 0]
            Q = sample[:, 1]
            amplitude = np.sqrt(I**2 + Q**2)
            amplitude = np.log10(amplitude + 1e-10)
            phase = np.arctan2(Q, I)
            
            amp_norm = self._min_max_normalize(amplitude)
            phase_norm = self._min_max_normalize(phase)
            processed_sample = np.stack([amp_norm, phase_norm], axis=-1)  # [seq_length, 2]
            processed.append(processed_sample)
        
        return np.array(processed, dtype=np.float32)

    def _min_max_normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val < 1e-10:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]  # [seq_length, 2]
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        seq_length, input_dim = sample_tensor.shape

        # 创建mask（最后一个元素强制mask）
        mask_matrix = torch.ones(seq_length, dtype=torch.float32)
        num_masked = int(seq_length * self.mask_ratio)
        # 从前面的元素中随机选mask（避开最后一个，最后一个单独处理）
        mask_indices = torch.randperm(seq_length - 1)[:num_masked]
        mask_matrix[mask_indices] = 0
        mask_matrix[-1] = 0  # 最后一个元素强制mask

        # 应用mask
        masked_input = sample_tensor * mask_matrix.unsqueeze(-1)  # [seq_length, 2]

        # 提取pre_label：序列的最后一个元素（形状[input_dim]，即[2]）
        pre_label = sample_tensor[-1]  # [2]

        return {
            'masked_input': masked_input,
            'mask_matrix': mask_matrix,
            'label': sample_tensor,  # 完整序列（用于全序列重构）
            'pre_label': pre_label   # 最后一个元素（新增）
        }

def process_pretrain_data(input_pkl_path, output_pkl_path, mask_ratio=0.3, batch_size=32):
    # 1. 读取输入pkl（原始数据，包含'value'、'label'、'snr'）
    print(f"正在读取输入文件：{input_pkl_path}")
    with open(input_pkl_path, 'rb') as f:
        input_data = pickle.load(f)
    
    # 提取原始数据（保留标签和SNR用于监督训练）
    raw_iq_data = input_data['value']  # [num_samples, 2, 1024]
    raw_labels = input_data['label']   # [num_samples]（原始标签）
    raw_snrs = input_data['snr']       # [num_samples]（原始SNR）
    
    # 转换IQ数据形状为[num_samples, 1024, 2]（适配幅相处理）
    raw_iq_data = raw_iq_data.transpose(0, 2, 1)
    print(f"输入数据形状：{raw_iq_data.shape}（样本数×序列长度×IQ通道）")

    # 2. 处理数据（幅相转换+mask+pre_label提取）
    dataset = PreTrainDataset(raw_iq_data, mask_ratio=mask_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. 收集处理结果（包含pre_label）
    print("开始处理数据...")
    processed_data = {
        # 监督训练所需（幅相数据替代原始IQ）
        'value': [],          # 幅相原始数据 [num_samples, 1024, 2]
        'label': raw_labels,  # 原始标签（不变）
        'snr': raw_snrs,      # 原始SNR（不变）
        # 自监督预训练所需
        'masked_input': [],   # 带mask的幅相数据 [num_samples, 1024, 2]
        'mask_matrix': [],    # mask标记 [num_samples, 1024]
        'pre_label': []       # 每个序列的最后一个元素 [num_samples, 2]
    }

    for batch in tqdm(dataloader, desc="批量处理"):
        processed_data['value'].append(batch['label'].cpu().numpy())  # 幅相原始数据
        processed_data['masked_input'].append(batch['masked_input'].cpu().numpy())
        processed_data['mask_matrix'].append(batch['mask_matrix'].cpu().numpy())
        processed_data['pre_label'].append(batch['pre_label'].cpu().numpy())  # 新增pre_label

    # 4. 合并批量数据
    processed_data['value'] = np.concatenate(processed_data['value'], axis=0)
    processed_data['masked_input'] = np.concatenate(processed_data['masked_input'], axis=0)
    processed_data['mask_matrix'] = np.concatenate(processed_data['mask_matrix'], axis=0)
    processed_data['pre_label'] = np.concatenate(processed_data['pre_label'], axis=0)  # 合并pre_label

    # 5. 保存包含pre_label的pkl
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"处理完成！输出文件已保存至：{output_pkl_path}")
    print(f"输出数据包含键：{processed_data.keys()}")
    print(f"各键形状：")
    print(f"- value: {processed_data['value'].shape}")
    print(f"- label: {processed_data['label'].shape}")
    print(f"- snr: {processed_data['snr'].shape}")
    print(f"- masked_input: {processed_data['masked_input'].shape}")
    print(f"- mask_matrix: {processed_data['mask_matrix'].shape}")
    print(f"- pre_label: {processed_data['pre_label'].shape}")  # 确认pre_label形状

if __name__ == "__main__":
    # 配置参数
    INPUT_PKL_PATH = "test_signal_dataset_20251108_132343"  # 输入原始pkl路径
    OUTPUT_PKL_PATH = "pretrain_with_prelabel.pkl"  # 输出包含pre_label的pkl路径
    MASK_RATIO = 0.3
    BATCH_SIZE = 32

    process_pretrain_data(
        input_pkl_path=INPUT_PKL_PATH,
        output_pkl_path=OUTPUT_PKL_PATH,
        mask_ratio=MASK_RATIO,
        batch_size=BATCH_SIZE
    )