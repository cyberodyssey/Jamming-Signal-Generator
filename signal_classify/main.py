import numpy as np
import sys
import os
import scipy.io as sio
import time
import random
from datetime import datetime
from scipy.signal import stft
from scipy.fftpack import fftshift,fft
from matplotlib import pyplot as plt
import pickle
import h5py
from sklearn.model_selection import train_test_split
from scipy import signal
def plot_spectrogram(signal,fs,save=False):
    """使用 STFT 绘制时频图"""
    # 计算 STFT
    num = 128

    fre, ts, zxx = stft(signal, fs=fs, nperseg=num, noverlap=num - 1, nfft=num,
                        return_onesided=False, window='hann')
    Fs = np.linspace(-fs/ 2, fs / 2, num)
    Z = np.abs(fftshift(zxx, 0))  # 出现在零频
    # Z = np.sqrt(Z)
    # 绘图
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.clf()
    # plt.pcolormesh(ts * 1e6, Fs * 1e-6, Z, cmap=parula)
    plt.pcolormesh(ts * 1e6, Fs * 1e-6, Z)
    plt.xlabel('Time/μs')
    plt.ylabel('Frequency/MHz')
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().set_position([0, 0, 1, 1])
    # plt.show()
    if save==True:
        plt.savefig(os.path.join('spectrogram', f'spectrogram_{datetime.now().strftime("%H%M%S%f")[:-3]}.png'))
    else:
        plt.show()

    plt.close()
    # fig = plt.gcf()
    # fig.canvas.draw()
    # ax.set_xlabel("Time (µs)")
    # ax.set_ylabel("Frequency (MHz)")
    # # ax.set_title("STFT Spectrogram")
    # ax.grid()


class SignalGenerator:
    def __init__(self,template):
        self.output_length = None

        self.fs=template["fs"]
        self.time_width=template["time_width"]
        self.band_width=template["band_width"]
        self.output_length=template["output_length"]
        self.snr = template["snr"]
        self.signal = np.zeros((self.output_length),dtype=np.complex64)
        self.sig_length =  int(self.fs*self.time_width)

        # print(self.sig_length)


    def generate_lfm(self):
        B = self.band_width  # 带宽 (Hz)
        Fs = self.fs  # 采样率 (Hz)
        Ts = 1 / Fs  # 采样间隔 (s)
        Tp = self.sig_length * Ts  # 脉冲宽度 (s)
        k = B / Tp  # 调频斜率 (Hz/s)
        # 时间序列，从 0 到 Tp
        t = np.linspace(0, Tp, self.sig_length)
        # 调频信号的频率范围从 -B/2 到 B/2
        # 信号表达式： s(t) = exp(1j * 2 * pi * (f0 * t + (k / 2) * t^2))
        choice = np.random.randint(2)
        if choice == 0:
            # 线性上升：频率随时间从 -B/2 增加到 B/2
            # 相位公式：φ(t) = 2π * [ (-B/2)*t + (k/2)*t² ]
            # 推导：瞬时频率 f(t) = dφ/dt / (2π) = -B/2 + k*t，当t=Tp时，f=B/2（因k=B/Tp）
            phase = 2 * np.pi * ((-B / 2) * t + (k / 2) * t ** 2)
        else:
            # 线性下降：频率随时间从 B/2 降低到 -B/2
            # 相位公式：φ(t) = 2π * [ (B/2)*t - (k/2)*t² ]
            # 推导：瞬时频率 f(t) = dφ/dt / (2π) = B/2 - k*t，当t=Tp时，f=-B/2
            phase = 2 * np.pi * ((B / 2) * t - (k / 2) * t ** 2)
        generate_signal = np.exp(1j * phase)
        self.signal = generate_signal[:self.output_length]
        print(self.signal.shape)
        return generate_signal[:self.output_length]

    def generate_nlfm(self, n=2):  # n为多项式阶数，n≥2实现非线性
        B = self.band_width  # 带宽 (Hz)
        Fs = self.fs  # 采样率 (Hz)
        Ts = 1 / Fs  # 采样间隔 (s)
        Tp = self.sig_length * Ts  # 脉冲宽度 (s)
        t = np.linspace(0, Tp, self.sig_length)  # 时间序列
        # 归一化时间x ∈ [0,1]
        x = t / Tp
        # 瞬时相位（基于n次多项式的NLFM）
        choice = np.random.randint(4)
        if choice == 0:
            # 1. 上升：f(x)从-B/2 → B/2，多项式P(x)=xⁿ （非线性递增）
            # 频率公式：f(x) = -B/2 + B·xⁿ
            # 相位公式：φ = 2π·Tp·[ -B/2·x + B/(n+1)·xⁿ⁺¹ ]
            phase = 2 * np.pi * Tp * ((-B / 2) * x + (B / (n + 1)) * x ** (n + 1))
            trend = f"上升（xⁿ）：f从-B/2→B/2，n={n}"

        elif choice == 1:
            # 2. 上升：f(x)从-B/2 → B/2，多项式P(x)=1 - (1-x)ⁿ （先慢后快）
            # 频率公式：f(x) = -B/2 + B·[1 - (1-x)ⁿ]
            phase = 2 * np.pi * Tp * ((-B / 2) * x + B * (x + (1 - x) ** (n + 1) / (n + 1) - 1 / (n + 1)))
            trend = f"上升（1-(1-x)ⁿ）：f从-B/2→B/2，n={n}"

        elif choice == 2:
            # 3. 下降：f(x)从B/2 → -B/2，多项式P(x)=1 - xⁿ （非线性递减）
            # 频率公式：f(x) = B/2 - B·xⁿ
            phase = 2 * np.pi * Tp * ((B / 2) * x - (B / (n + 1)) * x ** (n + 1))
            trend = f"下降（1-xⁿ）：f从B/2→-B/2，n={n}"

        elif choice == 3:
            # 4. 下降：f(x)从B/2 → -B/2，多项式P(x)=(1-x)ⁿ （先快后慢）
            # 频率公式：f(x) = B/2 - B·(1-x)ⁿ
            phase = 2 * np.pi * Tp * ((B / 2) * x - B * ((1 - (1 - x) ** (n + 1)) / (n + 1)))
            trend = f"下降（(1-x)ⁿ）：f从B/2→-B/2，n={n}"

        # -1+2 x^3
        # 生成复信号
        generate_signal = np.exp(1j * phase)
        self.signal = generate_signal[:self.output_length]
        print(self.signal.shape)
        return generate_signal[:self.output_length]

    def generate_isrj_from_lfm_signal(self, segment_num=np.random.choice([10,20]), zero_percent=0.5):
        """
        生成带随机段置零的ISRJ信号（基于LFM信号）
        参数：
            segment_num: int，信号切分的总段数（默认10段）
            zero_percent: float，需置零的段百分比（范围0-1，默认30%）
        返回：
            isrj_sig: np.ndarray，处理后的ISRJ信号
        """
        # 1. 生成原始LFM信号
        ori_sig = self.generate_lfm()
        sig_length = len(ori_sig)  # 原始信号总长度

        # 2. 计算每段长度，确保均匀切分（若总长度不能整除，最后一段补全）
        segment_len = sig_length // segment_num  # 基础段长度
        remain = sig_length % segment_num  # 剩余未分配的点数（补到最后一段）

        # 3. 切分信号为子段列表
        segments = []
        start = 0
        for i in range(segment_num):
            # 最后一段需包含剩余点数，其他段用基础长度
            end = start + segment_len + (remain if i == segment_num - 1 else 0)
            segments.append(ori_sig[start:end])
            start = end

        # 4. 计算需置零的段数，随机选择置零段
        zero_percent = np.random.choice([0.7,0.6,0.5,0.4])
        zero_segment_count = int(segment_num * zero_percent)  # 需置零的段数（取整）
        zero_segment_indices = random.sample(range(segment_num), zero_segment_count)  # 随机选段的索引

        # 5. 将选中的段置零（复信号置零需实部虚部均为0，直接赋值0即可）
        for idx in zero_segment_indices:
            segments[idx] = np.zeros_like(segments[idx])  # 保持与原段相同的形状和数据类型

        # 6. 拼接所有子段，得到最终ISRJ信号
        isrj_sig = np.concatenate(segments)
        self.signal = isrj_sig
        return isrj_sig

    def generate_isrj_from_nlfm_signal(self, segment_num=np.random.choice([10,20]), zero_percent=0.2):
        """
        生成带随机段置零的ISRJ信号（基于LFM信号）
        参数：
            segment_num: int，信号切分的总段数（默认10段）
            zero_percent: float，需置零的段百分比（范围0-1，默认30%）
        返回：
            isrj_sig: np.ndarray，处理后的ISRJ信号
        """
        # 1. 生成原始LFM信号
        ori_sig = self.generate_nlfm()
        sig_length = len(ori_sig)  # 原始信号总长度

        # 2. 计算每段长度，确保均匀切分（若总长度不能整除，最后一段补全）
        segment_len = sig_length // segment_num  # 基础段长度
        remain = sig_length % segment_num  # 剩余未分配的点数（补到最后一段）

        # 3. 切分信号为子段列表
        segments = []
        start = 0
        for i in range(segment_num):
            # 最后一段需包含剩余点数，其他段用基础长度
            end = start + segment_len + (remain if i == segment_num - 1 else 0)
            segments.append(ori_sig[start:end])
            start = end

        # 4. 计算需置零的段数，随机选择置零段
        zero_percent = np.random.choice([0.7, 0.6, 0.5, 0.4])
        zero_segment_count = int(segment_num * zero_percent)  # 需置零的段数（取整）
        zero_segment_indices = random.sample(range(segment_num), zero_segment_count)  # 随机选段的索引

        # 5. 将选中的段置零（复信号置零需实部虚部均为0，直接赋值0即可）
        for idx in zero_segment_indices:
            segments[idx] = np.zeros_like(segments[idx])  # 保持与原段相同的形状和数据类型

        # 6. 拼接所有子段，得到最终ISRJ信号
        isrj_sig = np.concatenate(segments)
        self.signal = isrj_sig
        return isrj_sig


    def save_signal(self):
        current_time = datetime.now().strftime("%H%M%S%f")[:-3]  # 当前时分秒
        file_prefix = f"{current_time}"
        sio.savemat(file_prefix, {"signal": self.signal})
        self.signal.clear()

    def add_white_noise(self, target_snr):
        """响应 open_noise 按钮事件，生成并添加白噪声"""
        snr_db = target_snr
        # LFM 信号的能量（振幅为 1）
        lfm_amplitude = 1.0

        # 根据 SNR 值计算噪声功率
        snr_linear = 10 ** (snr_db / 10)  # 将 SNR dB 转换为线性值
        noise_power = 1 / snr_linear  # 噪声功率

        # 白噪声参数
        noise_length = self.output_length  # 白噪声长度（与整体信号长度一致）
        start_point = 0  # 起始点，默认 0
        # print(len(self.signal))
        # 生成白噪声
        self.signal += self.generate_white_noise(noise_length, noise_power)
        print(f"白噪声已成功添加，SNR: {snr_db:.2f} dB，噪声功率: {noise_power:.5f}")
        # except ValueError:
        #     print("SNR 输入无效，请输入有效的数字。")

    def generate_white_noise(self, length, power):
        """生成白噪声信号"""
        # 根据功率生成复数白噪声
        noise = (np.random.randn(length) + 1j * np.random.randn(length)) * np.sqrt(power / 2)
        return noise


    def generate_fixed_jamming(self):
        B = self.band_width  # 带宽 (Hz)
        Fs = self.fs  # 采样率 (Hz)
        Ts = 1 / Fs  # 采样间隔 (s)
        Tp = self.sig_length * Ts  # 脉冲宽度 (s)
        t = np.linspace(0, Tp, self.sig_length)  # 时间序列
        # 归一化时间x ∈ [0,1]
        x = t / Tp
        # 生成复信号
        jamming_sig = np.exp(1j * 2 * np.pi * (-B / 2 * t))
        self.signal += jamming_sig

    def generate_laser_jamming(self):
        import numpy as np

        # 已有参数
        B = self.band_width  # 带宽 (Hz)
        Fs = self.fs  # 采样率 (Hz)
        Ts = 1 / Fs  # 采样间隔 (s)
        Tp = self.sig_length * Ts  # 脉冲宽度 (s)
        total_samples = self.sig_length  # 总采样点数

        # 1. 随机选择时间分段数（4、8、16中随机选）
        time_segment_options = [4, 6, 8, 10, 12, 14, 16]
        time_segments = np.random.choice(time_segment_options)  # 时间分段数（如4段）

        # 2. 随机生成频段池（数量为4、8、16中随机选，作为候选频段）
        freq_pool_options = [4, 6, 8, 10, 12, 14, 16]
        freq_pool_size = np.random.choice(freq_pool_options)  # 频段池大小（如16个频段）
        freq_pool = np.random.uniform(low=-B / 2, high=B / 2, size=freq_pool_size)  # 生成频段池

        # 3. 为每个时间段从频段池中随机选1个频段（允许重复）
        # 例如：4个时间段，每个都从16个频段中随机挑1个
        segment_freqs = np.random.choice(freq_pool, size=time_segments)

        # 4. 时间分段：总采样点按时间段数分配（每段至少1个采样点）
        time_segment_samples = []
        base_samples = total_samples // time_segments  # 每段基础采样数
        remainder = total_samples % time_segments  # 剩余采样数（补到前remainder段）
        for i in range(time_segments):
            if i < remainder:
                time_segment_samples.append(base_samples + 1)  # 前remainder段多1个采样点
            else:
                time_segment_samples.append(base_samples)

        # 5. 生成最终信号
        t = np.linspace(0, Tp, total_samples)  # 总时间序列
        jamming_sig = np.zeros_like(t, dtype=np.complex64)  # 初始化复信号
        current_sample = 0  # 当前采样点索引

        for seg_idx in range(time_segments):
            # 当前时间段的参数
            n_samples = time_segment_samples[seg_idx]  # 本时段采样数
            freq = segment_freqs[seg_idx]  # 从频段池随机选中的频段

            # 计算本时段的起止采样点
            start_sample = current_sample
            end_sample = start_sample + n_samples
            end_sample = min(end_sample, total_samples)  # 避免超出总长度

            # 生成本时段的复信号（单频信号）
            t_segment = t[start_sample:end_sample]
            jamming_sig[start_sample:end_sample] = 1 * np.exp(1j * 2 * np.pi * freq * t_segment)

            # 更新当前采样点索引
            current_sample = end_sample

        # 叠加到原始信号
        self.signal += jamming_sig

    def get_sig(self):
        return self.signal



def generate_amc_signal_dataset(train_save_path, test_save_path, per_type_per_snr=400, test_size=0.1,plot=False,save=False):
    """
    生成信号数据集并按9:1划分，保存为.pkl文件（字典格式）
    输出字典结构：
    {
        "value": (N, 2, 1024) float32数组,
        "label": (N,) int32数组,
        "snr": (N,) int32数组
    }
    """
    # 1. 基础参数定义
    fs_list = [16e6, 32e6, 64e6]
    snr_list = np.arange(-20, 19, 2).astype(int)  # -20到18，步长2
    signal_type_map = {
        "lfm": 0, "nlfm": 1, "isrj-lfm": 2,
        "isrj-nlfm": 3, "fixed": 4, "laser": 5
    }
    signal_types = list(signal_type_map.keys())

    # 2. 计算总样本数并初始化数组
    total_samples = len(signal_types) * len(snr_list) * per_type_per_snr
    values = np.empty((total_samples, 2, 1024), dtype=np.float32)  # 存储所有信号的IQ数据
    labels = np.empty(total_samples, dtype=np.int32)                # 存储所有标签
    snrs = np.empty(total_samples, dtype=np.int32)                  # 存储所有SNR值

    print(f"开始生成数据集，总样本数：{total_samples}")
    sample_idx = 0  # 数组索引计数器

    # 3. 生成信号样本
    for sig_type in signal_types:
        label = signal_type_map[sig_type]
        print(f"\n===== 生成信号类型：{sig_type}（标签：{label}） =====")
        for snr in snr_list:
            snr_int = int(snr)
            print(f"  生成SNR={snr_int}dB的样本（{per_type_per_snr}个）...")
            for _ in range(per_type_per_snr):
                # 随机参数设置
                fs = random.choice(fs_list)
                time_width = 1024 / fs  # 固定1024采样点
                percentages = np.arange(0.25, 0.51, 0.05)  # 25%到50%带宽
                band_widths = (percentages * fs).astype(int).tolist()
                band_width = random.choice(band_widths)

                # 初始化信号生成器
                template = {
                    "fs": fs,
                    "time_width": time_width,
                    "band_width": band_width,
                    "output_length": 1024,
                    "snr": snr_int
                }
                gen = SignalGenerator(template)

                # 生成对应类型的信号（修复信号生成逻辑）
                if sig_type == "lfm":
                    gen.generate_lfm()
                elif sig_type == "nlfm":
                    gen.generate_nlfm()
                elif sig_type == "isrj-lfm":
                    gen.generate_isrj_from_lfm_signal()
                elif sig_type == "isrj-nlfm":
                    gen.generate_isrj_from_nlfm_signal()
                elif sig_type == "fixed":
                    gen.signal = np.zeros(1024, dtype=np.complex64)  # 避免叠加旧信号
                    gen.generate_fixed_jamming()
                elif sig_type == "laser":
                    gen.signal = np.zeros(1024, dtype=np.complex64)  # 避免叠加旧信号
                    gen.generate_laser_jamming()

                # 添加噪声并转换格式
                gen.add_white_noise(target_snr=snr_int)
                sig_data = gen.get_sig()  # 复数信号 (1024,)
                if plot==True:
                    plot_spectrogram(sig_data,fs,save=save)
                # 填充数组（IQ通道分离）
                values[sample_idx] = np.stack([sig_data.real, sig_data.imag], axis=0)
                labels[sample_idx] = label
                snrs[sample_idx] = snr_int

                sample_idx += 1
                if sample_idx % 1000 == 0:
                    print(f"  已生成{sample_idx}/{total_samples}个样本（{sample_idx/total_samples*100:.1f}%）")

    # 4. 分层划分训练集和测试集（保持类别分布）
    train_indices, test_indices = train_test_split(
        np.arange(total_samples),
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # 5. 构建AMCDataset所需的字典
    train_dict = {
        "value": values[train_indices],  # 训练集数据 (N_train, 2, 1024)
        "label": labels[train_indices],  # 训练集标签 (N_train,)
        "snr": snrs[train_indices]       # 训练集SNR (N_train,)
    }
    test_dict = {
        "value": values[test_indices],   # 测试集数据 (N_test, 2, 1024)
        "label": labels[test_indices],   # 测试集标签 (N_test,)
        "snr": snrs[test_indices]        # 测试集SNR (N_test,)
    }

    # 6. 保存为.pkl文件（关键：用pickle保存字典）
    with open(train_save_path, "wb") as f:
        pickle.dump(train_dict, f)
    print(f"\n训练集已保存至：{train_save_path}.pkl")
    print(f"训练集结构：value{train_dict['value'].shape}, label{train_dict['label'].shape}, snr{train_dict['snr'].shape}")

    with open(test_save_path, "wb") as f:
        pickle.dump(test_dict, f)
    print(f"测试集已保存至：{test_save_path}.pkl")
    print(f"测试集结构：value{test_dict['value'].shape}, label{test_dict['label'].shape}, snr{test_dict['snr'].shape}")



# ---------------------- 生成数据集 ----------------------
    """_summary_
    """
if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = f"signal_dataset_{current_time}"
    # per_type_per_snr为每类的个snr生成的样本数
    # plot为是否绘图
    # save为是否保存绘图
    generate_amc_signal_dataset(train_save_path=f"train_val_{save_file}", test_save_path=f"test_{save_file}", per_type_per_snr=2,plot=True,save=True)
# ---------------------- 手动生成数据集 ----------------------
# if __name__ == '__main__':
#     template ={
#         "fs": int(32*1e6),
#         "time_width": float(32*1e-6),
#         "band_width": int(16*1e6),
#         "output_length": int(1024),
#         "snr": float(10),
#     }

#     sig = SignalGenerator(template)
#     sig.generate_isrj_from_nlfm_signal()
#     # sig.add_white_noise()
#     # sig.generate_fixed_jamming()
#     # sig.generate_laser_jamming()
#     lfm_sig = sig.get_sig()
#     plot_spectrogram(lfm_sig,template["fs"])


#48000
#lfm/nlfm   isrj-lfm/nlfm  fix/laser
#-20~18
#
#   生成规则，fs从[16,32,64]*1e6中选择，output_length固定1024，time对应的用1024/[16,32,64]
#   band_width从fs的25%到50%划分步长5%，随机选择
#   先生成信号[lfm/nlfm/isrj-lfm/isrj-nlfm/fixed/laser]，dB白噪声在最后再加，从-20到18，步长为2dB，每dB生成40个
#   然后生成pkl，第一项为数据，第二项为信号类型，第三项为snr