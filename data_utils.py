import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from tools.log import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import cleaned_text_to_sequence
from config import config

"""Multi speaker version"""
import multiprocessing as mp
from multiprocessing import Pool, Manager
import math

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)
        self.empty_emo = torch.squeeze(
            torch.load("empty_emo.npy", map_location="cpu"), dim=1
        )

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
        self.valid_indices = []  # 新增：存储有效索引
        self._parallel_validate_data()  # 改为并行验证
        
        # 根据数据量选择验证模式
        data_size = len(self.audiopaths_sid_text)
        if data_size > 10000:
            # 大数据集：使用快速验证
            logger.info(f"数据集较大 ({data_size}条)，使用快速验证模式")
            problematic_data = self._quick_validate_data()
            
            # 对有问题数据再进行详细验证
            if problematic_data:
                logger.info("对问题数据进行详细验证...")
                self._detailed_validate_problematic(problematic_data)
        else:
            # 小数据集：使用并行验证
            logger.info(f"数据集较小 ({data_size}条)，使用并行验证模式")
            self._parallel_validate_data()
        
    def _validate_single_item(self, args):
        """单条数据验证函数，用于多进程"""
        idx, audiopath_sid_text, hparams_dict = args
        
        try:
            # 复制hparams设置
            from copy import deepcopy
            class SimpleHParams:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            
            hparams = SimpleHParams(hparams_dict)
            
            # 模拟get_text方法
            audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
            
            # 检查基础条件
            if hparams.min_text_len <= len(phones) and len(phones) <= hparams.max_text_len:
                phones_list = phones.split(" ")
                tone_list = [int(i) for i in tone.split(" ")]
                word2ph_list = [int(i) for i in word2ph.split(" ")]
                
                # 检查BERT文件是否存在
                bert_path = audiopath.replace(".wav", ".bert.pt")
                if not os.path.exists(bert_path):
                    return idx, False, "BERT文件不存在"
                
                # 检查音频文件是否存在
                if not os.path.exists(audiopath):
                    return idx, False, "音频文件不存在"
                
                # 加载BERT文件并检查长度
                try:
                    bert = torch.load(bert_path, map_location='cpu')
                    bert_len = bert.shape[-1]
                    
                    # 计算音素长度（考虑add_blank）
                    phone_len = len(phones_list)
                    if hparams.add_blank:
                        phone_len = phone_len * 2 + 1
                    
                    if bert_len == phone_len:
                        return idx, True, "成功"
                    else:
                        return idx, False, f"长度不匹配: bert={bert_len}, phone={phone_len}"
                        
                except Exception as e:
                    return idx, False, f"加载BERT失败: {str(e)}"
            else:
                return idx, False, f"音素长度超出范围: {len(phones)}"
                
        except Exception as e:
            return idx, False, f"验证异常: {str(e)}"
    
    def _parallel_validate_data(self):
        """并行验证所有数据"""
        logger.info("开始并行验证数据集...")
        
        # 准备hparams字典（可序列化）
        hparams_dict = {
            'min_text_len': self.min_text_len,
            'max_text_len': self.max_text_len,
            'add_blank': self.add_blank,
            'spk2id': self.spk_map,
        }
        
        # 准备任务参数
        tasks = [(idx, item, hparams_dict) 
                for idx, item in enumerate(self.audiopaths_sid_text)]
        
        # 确定进程数（使用CPU核心数）
        num_workers = min(mp.cpu_count(), 8)  # 最多8个进程
        logger.info(f"使用 {num_workers} 个进程进行验证")
        
        valid_indices = []
        problematic_data = []
        
        # 分批处理，避免内存问题
        batch_size = 1000
        num_batches = math.ceil(len(tasks) / batch_size)
        
        with Pool(processes=num_workers) as pool:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                logger.info(f"验证批次 {batch_idx + 1}/{num_batches} ({len(batch_tasks)}条数据)")
                
                # 并行处理批次
                results = pool.map(self._validate_single_item, batch_tasks)
                
                # 处理结果
                for idx, is_valid, message in results:
                    if is_valid:
                        valid_indices.append(idx)
                    else:
                        problematic_data.append((idx, message))
                
                # 显示进度
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    logger.info(f"  当前进度: 有效 {len(valid_indices)}, 无效 {len(problematic_data)}")
        
        # 按原始顺序排序
        valid_indices.sort()
        self.valid_indices = valid_indices
        
        # 输出统计信息
        logger.info(f"验证完成: 总数据 {len(self.audiopaths_sid_text)}, 有效数据 {len(self.valid_indices)}, 无效数据 {len(problematic_data)}")
        
        # 输出问题数据详情（前10条）
        if problematic_data:
            logger.warning("前10条问题数据:")
            for idx, message in problematic_data[:10]:
                item = self.audiopaths_sid_text[idx]
                logger.warning(f"  索引 {idx}: {item[0]} - {message}")
            
            # 保存问题数据到文件
            problem_file = os.path.join(os.path.dirname(self.audiopaths_sid_text[0][0]), "problematic_data.txt")
            with open(problem_file, 'w', encoding='utf-8') as f:
                for idx, message in problematic_data:
                    item = self.audiopaths_sid_text[idx]
                    f.write(f"{idx}|{item[0]}|{message}\n")
            logger.info(f"问题数据已保存到: {problem_file}")
    
    def __getitem__(self, index):
        # 使用验证后的索引
        actual_index = self.valid_indices[index]
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[actual_index])
    
    def __len__(self):
        # 返回有效数据长度
        return len(self.valid_indices)

    def _detailed_validate_problematic(self, problematic_data):
        """对问题数据进行详细验证"""
        logger.info("开始详细验证问题数据...")
        
        detailed_problematic = []
        recovered = 0
        
        for idx, reason in tqdm(problematic_data, desc="详细验证"):
            try:
                audiopath_sid_text = self.audiopaths_sid_text[idx]
                phones, spec, wav, sid, tone, language, bert, emo = \
                    self.get_audio_text_speaker_pair(audiopath_sid_text)
                
                if bert.shape[-1] == len(phones):
                    self.valid_indices.append(idx)
                    recovered += 1
                else:
                    detailed_problematic.append((idx, f"长度不匹配: bert={bert.shape[-1]}, phone={len(phones)}"))
                    
            except Exception as e:
                detailed_problematic.append((idx, f"详细验证失败: {str(e)}"))
        
        logger.info(f"详细验证完成: 恢复 {recovered} 条数据, 仍有问题 {len(detailed_problematic)} 条")
        
    #------------------
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text
        ):
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        emo = torch.squeeze(
            torch.load(audiopath.replace(".wav", ".emo.pt"), map_location="cpu"),
            dim=1,
        )
        return (phones, spec, wav, sid, tone, language, bert, emo)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert = torch.load(bert_path)
            
            # 调试信息
            if bert.shape[-1] != len(phone):
                logger.error(f"BERT与音素长度不匹配!")
                logger.error(f"  文件: {wav_path}")
                logger.error(f"  文本: {text}")
                logger.error(f"  音素: {phone}")
                logger.error(f"  BERT形状: {bert.shape}")
                logger.error(f"  音素长度: {len(phone)}")
                logger.error(f"  语言: {language_str}")
                
                # 尝试修复：截断较长的序列
                min_len = min(bert.shape[-1], len(phone))
                if min_len > 0:
                    bert = bert[:, :min_len]
                    phone = phone[:min_len]
                    tone = tone[:min_len]
                    language = language[:min_len]
                    logger.warning(f"  已截断到长度: {min_len}")
                else:
                    raise ValueError("截断后长度为0")
        except Exception as e:
            logger.error(f"加载BERT文件失败: {bert_path}, 错误: {e}")
            raise
        
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 2048, max_text_len)
        # en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        emo = torch.FloatTensor(len(batch), 512)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        # en_bert_padded.zero_()
        emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            # en_bert = row[7]
            # en_bert_padded[i, :, : en_bert.size(1)] = en_bert

            emo[i, :] = row[7]

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            bert_padded,
            emo,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        # 使用数据集的有效长度
        self.lengths = [dataset.lengths[i] for i in dataset.valid_indices]
        self.valid_indices = dataset.valid_indices  # 存储有效索引
        self.batch_size = batch_size
        self.boundaries = boundaries
        
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
    
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(self.valid_indices[i])  # 使用原始索引
        
        # ... 其余代码保持不变 ...

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
