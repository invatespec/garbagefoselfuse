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
import logging
# 创建全局logger，避免依赖外部logger
data_logger = logging.getLogger("data_utils")
data_logger.setLevel(logging.WARNING)

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
        self.valid_indices = []
        self.problematic_indices = set()  # 存储问题索引
        self._fast_validate_data()  # 使用快速验证

    def _fast_validate_data(self):
        """快速验证数据，使用多进程并行"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        data_logger.info(f"开始快速验证 {len(self.audiopaths_sid_text)} 条数据...")
        
        # 使用进程池并行验证
        n_workers = min(mp.cpu_count(), 8)  # 限制进程数
        batch_size = 100  # 每批验证的数量
        
        # 分批处理
        all_indices = list(range(len(self.audiopaths_sid_text)))
        batches = [all_indices[i:i+batch_size] for i in range(0, len(all_indices), batch_size)]
        
        self.valid_indices = []
        total_valid = 0
        total_problematic = 0
        
        # 单进程验证函数
        def validate_batch(batch_indices):
            valid_in_batch = []
            problematic_in_batch = []
            
            for idx in batch_indices:
                try:
                    # 快速检查：只需验证BERT和音素长度是否匹配
                    audiopath, sid, language, text, phones, tone, word2ph = self.audiopaths_sid_text[idx]
                    
                    # 1. 加载BERT文件
                    bert_path = audiopath.replace(".wav", ".bert.pt")
                    bert = torch.load(bert_path, map_location='cpu')
                    
                    # 2. 计算音素长度
                    phone_list = phones.split(" ")
                    if self.add_blank:
                        phone_len = len(phone_list) * 2 + 1
                    else:
                        phone_len = len(phone_list)
                    
                    # 3. 验证长度
                    if bert.shape[-1] == phone_len:
                        valid_in_batch.append(idx)
                    else:
                        problematic_in_batch.append((idx, bert.shape[-1], phone_len))
                        
                except Exception as e:
                    problematic_in_batch.append((idx, str(e)))
            
            return valid_in_batch, problematic_in_batch
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(validate_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in tqdm(as_completed(futures), total=len(batches), desc="验证数据"):
                valid_batch, problematic_batch = future.result()
                self.valid_indices.extend(valid_batch)
                total_valid += len(valid_batch)
                total_problematic += len(problematic_batch)
                
                # 记录问题数据
                for item in problematic_batch:
                    if len(item) == 3:
                        idx, bert_len, phone_len = item
                        self.problematic_indices.add(idx)
                        data_logger.debug(f"数据 {idx} 长度不匹配: bert_len={bert_len}, phone_len={phone_len}")
                    else:
                        idx, error = item
                        self.problematic_indices.add(idx)
                        data_logger.debug(f"数据 {idx} 加载失败: {error}")
        
        data_logger.info(f"验证完成: 有效数据 {total_valid} 条, 问题数据 {total_problematic} 条")
        
        # 如果没有有效数据，使用所有数据（兼容模式）
        if len(self.valid_indices) == 0:
            data_logger.warning("没有有效数据，使用所有数据（可能出错）")
            self.valid_indices = list(range(len(self.audiopaths_sid_text)))
    
    def __getitem__(self, index):
        # 使用验证后的索引
        actual_index = self.valid_indices[index]
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[actual_index])
    
    def __len__(self):
        # 返回有效数据长度
        return len(self.valid_indices)

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
