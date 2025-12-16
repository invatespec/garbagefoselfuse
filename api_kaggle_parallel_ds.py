import argparse
import os
import re
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
from text.LangSegmenter import LangSegmenter
from time import time as ttime
import torch
import torchaudio
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from peft import LoraConfig, get_peft_model
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
import config as global_config
import logging
import subprocess

import logging.config
import uvicorn

logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger("uvicorn")

# 模型实例访问记录，用于LRU淘汰
model_access_times = {}
# 当前已加载的模型计数
loaded_models_count = 0
# 长文本阈值
long_text_threshold = 70 

# --- 从这里开始，抄的 gradio_tunneling 的核心代码 ---
import atexit
import platform
import stat
import time
from pathlib import Path
from typing import List, Optional

import requests

VERSION = "0.2"
CURRENT_TUNNELS: List["Tunnel"] = []

machine = platform.machine()
if machine == "x86_64":
    machine = "amd64"

BINARY_REMOTE_NAME = f"frpc_{platform.system().lower()}_{machine.lower()}"
EXTENSION = ".exe" if os.name == "nt" else ""
BINARY_URL = f"https://cdn-media.huggingface.co/frpc-gradio-{VERSION}/{BINARY_REMOTE_NAME}{EXTENSION}"

BINARY_FILENAME = f"{BINARY_REMOTE_NAME}_v{VERSION}"
BINARY_FOLDER = Path(__file__).parent.absolute()
BINARY_PATH = f"{BINARY_FOLDER / BINARY_FILENAME}"

TUNNEL_TIMEOUT_SECONDS = 30
TUNNEL_ERROR_MESSAGE = (
    "Could not create share URL. "
    "Please check the appended log from frpc for more information:"
)

GRADIO_API_SERVER = "https://api.gradio.app/v2/tunnel-request"
GRADIO_SHARE_SERVER_ADDRESS = None


class Tunnel:
    def __init__(self, remote_host, remote_port, local_host, local_port, share_token):
        self.proc = None
        self.url = None
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_host = local_host
        self.local_port = local_port
        self.share_token = share_token

    @staticmethod
    def download_binary():
        if not Path(BINARY_PATH).exists():
            resp = requests.get(BINARY_URL)

            if resp.status_code == 403:
                raise OSError(
                    f"Cannot set up a share link as this platform is incompatible. Please "
                    f"create a GitHub issue with information about your platform: {platform.uname()}"
                )

            resp.raise_for_status()

            # Save file data to local copy
            with open(BINARY_PATH, "wb") as file:
                file.write(resp.content)
            st = os.stat(BINARY_PATH)
            os.chmod(BINARY_PATH, st.st_mode | stat.S_IEXEC)

    def start_tunnel(self) -> str:
        self.download_binary()
        self.url = self._start_tunnel(BINARY_PATH)
        return self.url

    def kill(self):
        if self.proc is not None:
            print(f"Killing tunnel {self.local_host}:{self.local_port} <> {self.url}")
            self.proc.terminate()
            self.proc = None

    def _start_tunnel(self, binary: str) -> str:
        CURRENT_TUNNELS.append(self)
        command = [
            binary,
            "http",
            "-n",
            self.share_token,
            "-l",
            str(self.local_port),
            "-i",
            self.local_host,
            "--uc",
            "--sd",
            "random",
            "--ue",
            "--server_addr",
            f"{self.remote_host}:{self.remote_port}",
            "--disable_log_color",
        ]
        self.proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        atexit.register(self.kill)
        return self._read_url_from_tunnel_stream()

    def _read_url_from_tunnel_stream(self) -> str:
        start_timestamp = time.time()

        log = []
        url = ""

        def _raise_tunnel_error():
            log_text = "\n".join(log)
            print(log_text, file=sys.stderr)
            raise ValueError(f"{TUNNEL_ERROR_MESSAGE}\n{log_text}")

        while url == "":
            # check for timeout and log
            if time.time() - start_timestamp >= TUNNEL_TIMEOUT_SECONDS:
                _raise_tunnel_error()

            assert self.proc is not None
            if self.proc.stdout is None:
                continue

            line = self.proc.stdout.readline()
            line = line.decode("utf-8")

            if line == "":
                continue

            log.append(line.strip())

            if "start proxy success" in line:
                result = re.search("start proxy success: (.+)\n", line)
                if result is None:
                    _raise_tunnel_error()
                else:
                    url = result.group(1)
            elif "login to server failed" in line:
                _raise_tunnel_error()

        return url


def setup_tunnel(
    local_host: str,
    local_port: int,
    share_token: str,
    share_server_address: Optional[str],
) -> str:
    share_server_address = (
        GRADIO_SHARE_SERVER_ADDRESS
        if share_server_address is None
        else share_server_address
    )
    if share_server_address is None:
        response = requests.get(GRADIO_API_SERVER)
        if not (response and response.status_code == 200):
            raise RuntimeError("Could not get share link from Gradio API Server.")
        payload = response.json()[0]
        remote_host, remote_port = payload["host"], int(payload["port"])
    else:
        remote_host, remote_port = share_server_address.split(":")
        remote_port = int(remote_port)
    try:
        tunnel = Tunnel(remote_host, remote_port, local_host, local_port, share_token)
        address = tunnel.start_tunnel()
        return address
    except Exception as e:
        raise RuntimeError(str(e)) from e
# --- 结束 ---

# ============ GPU环境检测 ============
def check_gpu_availability():
    """检测可用的GPU数量"""
    gpu_count = torch.cuda.device_count()
    logger.info(f"✅ 检测到 {gpu_count} 个GPU设备")
    
    if gpu_count == 0:
        logger.warning("❌ 未检测到GPU，将使用CPU模式")
        return 0, ["cpu"]
    elif gpu_count == 1:
        logger.info("🔧 单GPU环境，启用单卡优化模式")
        return 1, ["cuda:0"]
    else:
        logger.info(f"🚀 多GPU环境，启用并行模式")
        return gpu_count, [f"cuda:{i}" for i in range(gpu_count)]

# 检测GPU
GPU_COUNT, GPU_LIST = check_gpu_availability()
IS_MULTI_GPU = GPU_COUNT > 1

class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True


bigvgan_model = hifigan_model = sv_cn_model = None
def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def init_bigvgan(target_gpu="cuda:0"):
    # 声明全局变量
    global bigvgan_model
    
    from BigVGAN import bigvgan
    
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(target_gpu)
    else:
        bigvgan_model = bigvgan_model.to(target_gpu)

def init_hifigan(target_gpu="cuda:0"):
    # 声明全局变量
    global hifigan_model
    
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    if is_half == True:
        hifigan_model = hifigan_model.half().to(target_gpu)
    else:
        hifigan_model = hifigan_model.to(target_gpu)

from sv import SV
def init_sv_cn():
    global hifigan_model
    sv_cn_model = SV(device, is_half)


resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


from module.mel_processing import mel_spectrogram_torch

spec_min = -12
spec_max = 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            logger.info("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载")
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


# 2. 修改 Speaker 类，添加 gpt_path、sovits_path
class Speaker:
    def __init__(self, name, gpt=None, sovits=None, phones=None, bert=None, prompt=None, gpt_path=None, sovits_path=None, load_time=None,gpu0_gpt=None, gpu0_sovits=None, gpu1_gpt=None, gpu1_sovits=None,
last_used=None):
        self.name = name
        self.gpt = gpt
        self.sovits = sovits
        self.phones = phones
        self.bert = bert
        self.prompt = prompt
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        # 双GPU扩展字段
        self.gpu0_gpt = gpu0_gpt
        self.gpu0_sovits = gpu0_sovits
        self.gpu1_gpt = gpu1_gpt
        self.gpu1_sovits = gpu1_sovits
        self.last_used = last_used

class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps


from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new


def get_sovits_weights(sovits_path, target_gpu="cuda:0"):
    from config import pretrained_sovits_name
    path_sovits_v3 = pretrained_sovits_name["v3"]
    path_sovits_v4 = pretrained_sovits_name["v4"]
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)
    is_exist_s2gv4 = os.path.exists(path_sovits_v4)

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4

    if if_lora_v3 == True and is_exist == False:
        logger.info("SoVITS %s 底模缺失，无法加载相应 LoRA 权重" % model_version)

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    model_params_dict = vars(hps.model)
    if model_version not in {"v3", "v4"}:
        if "Pro" in model_version:
            hps.model.version = model_version
            if sv_cn_model == None:
                init_sv_cn()

        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict,
        )
        if model_version == "v3":
            init_bigvgan()
        if model_version == "v4":
            init_hifigan()

    model_version = hps.model.version
    logger.info(f"模型版本: {model_version}")
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(target_gpu)
    else:
        vq_model = vq_model.to(target_gpu)
    vq_model.eval()
    if if_lora_v3 == False:
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False)
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()

    sovits = Sovits(vq_model, hps)
    return sovits


class Gpt:
    def __init__(self, max_sec, t2s_model):
        self.max_sec = max_sec
        self.t2s_model = t2s_model


global hz
hz = 50


def get_gpt_weights(gpt_path, target_gpu="cuda:0"):
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = _safe_model_load(
     lambda: Text2SemanticLightningModule(config, "****", is_train=False)
    )
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(target_gpu)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # logger.info("Number of parameter: %.2fM" % (total / 1e6))

    gpt = Gpt(max_sec, t2s_model)
    return gpt


# 8. 修改 change_gpt_sovits_weights
def change_gpt_sovits_weights(gpt_path, sovits_path, speaker_id="default"):
    try:
        speaker_list[speaker_id] = Speaker(
            name=speaker_id,
            gpt=None,
            sovits=None,
            prompt=speaker_list.get(speaker_id, Speaker(name=speaker_id, gpt=None, sovits=None)).prompt or {
                "path": "D.wav",
                "text": "歌手でイプシロンのスーパースター…『傷つく誰かの心を守ることができたなら』って、アナタの作品だよね？",
                "prompt_language": "ja"
            },
            gpt_path=gpt_path,
            sovits_path=sovits_path
        )
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    except Exception as e:
        return JSONResponse({"code": 400, "message": str(e)}, status_code=400)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


from text import chinese


def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(torch.float16 if is_half == True else torch.float32), norm_text


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def pack_audio(audio_bytes, data, rate):
    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate)
    else:
        # wav无法流式, 先暂存raw
        audio_bytes = pack_raw(audio_bytes, data, rate)

    return audio_bytes


def pack_ogg(audio_bytes, data, rate):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(audio_bytes, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    import threading

    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return audio_bytes


def pack_raw(audio_bytes, data, rate):
    audio_bytes.write(data.tobytes())

    return audio_bytes


def pack_wav(audio_bytes, rate):
    if is_int32:
        data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int32)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format="WAV", subtype="PCM_32")
    else:
        data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format="WAV")
    return wav_bytes


def pack_aac(audio_bytes, data, rate):
    if is_int32:
        pcm = "s32le"
        bit_rate = "256k"
    else:
        pcm = "s16le"
        bit_rate = "128k"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            pcm,  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            bit_rate,  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)

    return audio_bytes


def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)

    return audio_bytes, audio_chunk


def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items) % 2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def unload_least_recently_used():
    """卸载最久未使用的模型（同时卸载该说话人在两个GPU上的实例）"""
    # 声明全局变量
    global loaded_models_count
    
    # 找到最久未使用的说话人
    if not speaker_list:
        return
    
    # 找到有模型加载且最久未使用的说话人
    candidates = []
    for speaker_id, speaker in speaker_list.items():
        if speaker.gpu0_gpt is not None or speaker.gpu1_gpt is not None:
            candidates.append((speaker_id, speaker.last_used or 0))
    
    if not candidates:
        return
    
    oldest_speaker_id = min(candidates, key=lambda x: x[1])[0]
    oldest_speaker = speaker_list[oldest_speaker_id]
    
    # 同时卸载该说话人在两个GPU上的实例
    if oldest_speaker.gpu0_gpt is not None:
        oldest_speaker.gpu0_gpt = None
        oldest_speaker.gpu0_sovits = None
        loaded_models_count -= 1
    
    if oldest_speaker.gpu1_gpt is not None:
        oldest_speaker.gpu1_gpt = None
        oldest_speaker.gpu1_sovits = None
        loaded_models_count -= 1
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    logger.info(f"🔄 已卸载最久未使用的说话人: {oldest_speaker_id}")

def ensure_model_loaded(speaker_id, gpu_index=None):
    """
    确保指定说话人的模型加载到指定的GPU上
    自动适配单GPU环境
    """
    from time import time as ttime
    
    # 必须在函数顶部声明所有要修改的全局变量
    global loaded_models_count, model_access_times
    
    if speaker_id not in speaker_list:
        raise ValueError(f"Speaker {speaker_id} not found")
    
    speaker = speaker_list[speaker_id]
    speaker.last_used = ttime()
    
    # 更新全局访问记录（用于LRU淘汰）
    model_access_times[speaker_id] = ttime()

    # 单GPU环境：只使用GPU 0
    if not IS_MULTI_GPU:
        gpu_index = 0
        target_gpu = "cuda:0"
        
        if speaker.gpu0_gpt is None:
            if loaded_models_count >= max_models:
                unload_least_recently_used()
            
            speaker.gpu0_gpt = get_gpt_weights(speaker.gpt_path, target_gpu)
            speaker.gpu0_sovits = get_sovits_weights(speaker.sovits_path, target_gpu)
            loaded_models_count += 1
        return
    
    # 如果指定了GPU索引
    if gpu_index is not None:
        # 检查GPU是否可用
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_index} 不可用，可用GPU数量: {torch.cuda.device_count()}")
        
        target_gpu = f"cuda:{gpu_index}"
        
        # 检查该GPU上的模型是否已加载
        if gpu_index == 0:
            if speaker.gpu0_gpt is None:
                # 检查内存是否充足
                free_mem, total_mem = check_gpu_memory(gpu_index)
                if free_mem < 1.0:  # 少于1GB空闲内存
                    logger.warning(f"GPU{gpu_index} 内存不足 ({free_mem:.2f}GB)，尝试清理...")
                    torch.cuda.empty_cache()
                    free_mem, _ = check_gpu_memory(gpu_index)
                    if free_mem < 0.5:  # 清理后仍少于500MB
                        # 卸载最久未使用的模型
                        unload_least_recently_used()
                
                # 检查是否需要卸载旧模型
                if loaded_models_count >= max_models * 2:
                    unload_least_recently_used()
                
                # 加载模型
                logger.info(f"正在加载模型到 GPU{gpu_index}...")
                speaker.gpu0_gpt = get_gpt_weights(speaker.gpt_path, target_gpu)
                speaker.gpu0_sovits = get_sovits_weights(speaker.sovits_path, target_gpu)
                loaded_models_count += 1
                logger.info(f"✅ 模型已加载到 GPU{gpu_index}")
        
        elif gpu_index == 1:
            if speaker.gpu1_gpt is None:
                # 检查内存是否充足
                free_mem, total_mem = check_gpu_memory(gpu_index)
                if free_mem < 1.0:  # 少于1GB空闲内存
                    logger.warning(f"GPU{gpu_index} 内存不足 ({free_mem:.2f}GB)，尝试清理...")
                    torch.cuda.empty_cache()
                    free_mem, _ = check_gpu_memory(gpu_index)
                    if free_mem < 0.5:  # 清理后仍少于500MB
                        # 卸载最久未使用的模型
                        unload_least_recently_used()
                
                # 检查是否需要卸载旧模型
                if loaded_models_count >= max_models * 2:
                    unload_least_recently_used()
                
                # 加载模型
                logger.info(f"正在加载模型到 GPU{gpu_index}...")
                speaker.gpu1_gpt = get_gpt_weights(speaker.gpt_path, target_gpu)
                speaker.gpu1_sovits = get_sovits_weights(speaker.sovits_path, target_gpu)
                loaded_models_count += 1
                logger.info(f"✅ 模型已加载到 GPU{gpu_index}")
    
    # 如果未指定GPU，确保至少一个GPU有模型
    else:
        if speaker.gpu0_gpt is None and speaker.gpu1_gpt is None:
            # 选择内存更充足的GPU
            free_mem_0, _ = check_gpu_memory(0)
            free_mem_1, _ = check_gpu_memory(1)
            
            selected_gpu = 0 if free_mem_0 >= free_mem_1 else 1
            logger.info(f"自动选择 GPU{selected_gpu} (空闲内存: {max(free_mem_0, free_mem_1):.2f}GB)")
            
            ensure_model_loaded(speaker_id, selected_gpu)

# 修改 get_tts_wav 函数，正确处理 get_spepc 的返回值
def get_tts_wav(
    refer_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    top_k=15,
    top_p=0.6,
    temperature=0.6,
    speed=1.0,
    inp_refs=None,
    sample_steps=32,
    if_sr=False,
    spk="default"
):
    from time import time as ttime
    import asyncio
    import concurrent.futures
    import numpy as np

    # 根据GPU环境决定是否启用并行
    if IS_MULTI_GPU:
        is_long, text_segments = split_long_text(text, long_text_threshold)
    else:
        # 单GPU环境：长文本也使用单卡处理
        is_long = False
        text_segments = [text]
        logger.info("🔧 单GPU环境，禁用并行处理")
    
    # 更新模型访问时间
    if spk in speaker_list:
        speaker_list[spk].last_used = ttime()
    
    # 2. 获取说话人信息
    if spk not in speaker_list:
        raise ValueError(f"Speaker {spk} not found")
    
    speaker = speaker_list[spk]
    
    # 3. 根据是否长文本选择处理方式
    if is_long:
        # ============ 长文本并行处理 ============
        logger.info(f"📖 长文本检测 ({len(text)}字 > {long_text_threshold})，启用双GPU并行处理")
        
        # 3.1 确保两个GPU上都有模型
        ensure_model_loaded(spk, 0)
        ensure_model_loaded(spk, 1)
        
        # 3.2 并行处理两个文本片段
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 准备两个任务
            futures = []
            for i, segment in enumerate(text_segments):
                if not segment.strip():  # 跳过空片段
                    continue
                    
                target_gpu = i % 2  # 0或1
                logger.info(f"  GPU{target_gpu} 处理片段 {i+1}: {segment[:30]}...")
                
                # 提交任务到线程池
                future = executor.submit(
                    _process_single_segment,
                    text_segment=segment,
                    refer_wav_path=refer_wav_path,
                    prompt_text=prompt_text,
                    prompt_language=prompt_language,
                    text_language=text_language,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    speed=speed,
                    inp_refs=inp_refs,
                    sample_steps=sample_steps,
                    if_sr=if_sr,
                    spk=spk,
                    gpu_index=target_gpu
                )
                futures.append((i, future))
            
            # 3.3 收集结果并保持顺序
            segment_results = []
            for seg_idx, future in sorted(futures, key=lambda x: x[0]):
                try:
                    # 获取音频数据：audio_array, sample_rate
                    audio_data, sr = future.result(timeout=120)  # 120秒超时
                    segment_results.append((seg_idx, audio_data, sr))
                except concurrent.futures.TimeoutError:
                    logger.error(f"❌ 片段 {seg_idx} 处理超时")
                    raise
                except Exception as e:
                    logger.error(f"❌ 片段 {seg_idx} 处理失败: {e}")
                    raise
        
        # 3.4 合并音频片段
        if not segment_results:
            raise ValueError("没有有效的音频片段生成")
            
        # 按原始顺序排序
        segment_results.sort(key=lambda x: x[0])
        
        # 获取第一个片段的采样率作为参考
        first_audio, first_sr = segment_results[0][1], segment_results[0][2]
        
        # 合并所有音频片段
        all_audio_segments = []
        for seg_idx, (_, audio_data, sr) in enumerate(segment_results):
            # 确保采样率一致
            if sr != first_sr:
                audio_data = _resample_audio_if_needed(audio_data, sr, first_sr)
            
            all_audio_segments.append(audio_data)
            
            # 在片段间添加静音（除最后一个片段外）
            if seg_idx < len(segment_results) - 1:
                # 创建0.3秒的静音
                silence = np.zeros(int(first_sr * 0.3), dtype=audio_data.dtype)
                all_audio_segments.append(silence)
        
        # 拼接所有音频
        combined_audio = np.concatenate(all_audio_segments, axis=0)
        
        # 最终归一化
        max_audio = np.abs(combined_audio).max()
        if max_audio > 1.0:
            combined_audio = combined_audio / max_audio
        
        final_audio = combined_audio
        final_sr = first_sr
        
    else:
        # ============ 短文本单GPU处理 ============
        logger.info(f"📝 短文本检测 ({len(text)}字 ≤ {long_text_threshold})，使用单GPU处理")
        
        # 确定使用哪个GPU
        selected_gpu = 0
        # 检查哪个GPU有模型
        if speaker.gpu1_gpt is not None and speaker.gpu0_gpt is None:
            selected_gpu = 1
        elif speaker.gpu0_gpt is not None:
            selected_gpu = 0
        else:
            # 两个都没有，随机选一个
            import random
            selected_gpu = random.choice([0, 1])
        
        # 确保模型已加载
        ensure_model_loaded(spk, selected_gpu)
        logger.info(f"  使用 GPU{selected_gpu} 处理短文本")
        
        # 处理单个文本片段
        final_audio, final_sr = _process_single_segment(
            text_segment=text,
            refer_wav_path=refer_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            inp_refs=inp_refs,
            sample_steps=sample_steps,
            if_sr=if_sr,
            spk=spk,
            gpu_index=selected_gpu
        )
    
    # 4. 包装音频为字节流
    all_audio_bytes = BytesIO()
    
    # 根据数据类型编码音频
    if is_int32:
        audio_data_int = (final_audio * 2147483647).astype(np.int32)
    else:
        audio_data_int = (final_audio * 32767).astype(np.int16)
    
    # 根据媒体类型打包音频
    if media_type == "wav":
        audio_bytes = pack_wav(audio_data_int, final_sr)
    else:
        audio_bytes = pack_audio(all_audio_bytes, audio_data_int, final_sr)
    
    # 5. 返回音频数据
    if stream_mode == "normal":
        # 流式返回
        audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
        yield audio_chunk
    else:
        # 一次性返回
        yield audio_bytes.getvalue()

def _process_single_segment(text_segment, refer_wav_path, prompt_text, prompt_language,
                           text_language, top_k, top_p, temperature, speed,
                           inp_refs, sample_steps, if_sr, spk, gpu_index):
    """
    处理单个文本片段（内部函数，用于并行处理）
    返回: (audio_data, sampling_rate)
    """
    try:
        global bigvgan_model, hifigan_model, sv_cn_model
        
        # 1. 获取目标GPU
        target_gpu = f"cuda:{gpu_index}"
        
        # 2. 确保模型已加载到目标GPU
        ensure_model_loaded(spk, gpu_index)
        
        # 3. 获取对应GPU上的模型实例
        speaker = speaker_list[spk]
        if gpu_index == 0:
            model_instance = {
                "gpt": speaker.gpu0_gpt,
                "sovits": speaker.gpu0_sovits
            }
        elif gpu_index == 1:
            model_instance = {
                "gpt": speaker.gpu1_gpt,
                "sovits": speaker.gpu1_sovits
            }
        else:
            raise ValueError(f"Invalid GPU index: {gpu_index}")
        
        if model_instance["gpt"] is None or model_instance["sovits"] is None:
            raise ValueError(f"Model not loaded on GPU {gpu_index} for speaker {spk}")
        
        # 4. 获取模型实例和配置
        infer_sovits = model_instance["sovits"]
        infer_gpt = model_instance["gpt"]
        
        vq_model = infer_sovits.vq_model
        hps = infer_sovits.hps
        version = vq_model.version
        
        t2s_model = infer_gpt.t2s_model
        max_sec = infer_gpt.max_sec
        
        # 5. 参数调整（与原始函数相同）
        if version == "v3":
            if sample_steps not in [4, 8, 16, 32, 64, 128]:
                sample_steps = 32
        elif version == "v4":
            if sample_steps not in [4, 8, 16, 32]:
                sample_steps = 8
        
        if if_sr and version != "v3":
            if_sr = False
        
        # 6. 准备参考音频（与原始函数相同，但指定目标GPU）
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        
        prompt_language, text_segment = prompt_language, text_segment.strip("\n")
        dtype = torch.float16 if is_half == True else torch.float32
        
        # 创建静音片段（与原始函数相同）
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
        zero_wav_torch = torch.from_numpy(zero_wav)
        
        # 7. 参考音频处理（与原始函数相同，但指定目标GPU）
        with torch.no_grad():
            wav16k, sr = librosa.load(refer_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            
            if is_half == True:
                wav16k = wav16k.half().to(target_gpu)
                zero_wav_torch = zero_wav_torch.half().to(target_gpu)
            else:
                wav16k = wav16k.to(target_gpu)
                zero_wav_torch = zero_wav_torch.to(target_gpu)
            
            wav16k = torch.cat([wav16k, zero_wav_torch])
            
            # SSL模型处理
            ssl_device = None
            try:
                if hasattr(ssl_model, 'parameters'):
                    ssl_device = next(ssl_model.parameters()).device
                elif hasattr(ssl_model, 'device'):
                    ssl_device = ssl_model.device
            except Exception as e:
                ssl_device = torch.device("cpu")
            
            if ssl_device.type != "cpu":
                wav16k_for_ssl = wav16k.to(ssl_device)
                ssl_content = ssl_model.model(wav16k_for_ssl.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                ssl_content = ssl_content.to(target_gpu)
            else:
                wav16k_cpu = wav16k.cpu()
                ssl_content = ssl_model.model(wav16k_cpu.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                if is_half == True:
                    ssl_content = ssl_content.half()
                ssl_content = ssl_content.to(target_gpu)
            
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(target_gpu)
            
            # 获取参考频谱（根据模型版本）
            is_v2pro = version in {"v2Pro", "v2ProPlus"}
            if version not in {"v3", "v4"}:
                refers = []
                if is_v2pro:
                    sv_emb = []
                    if sv_cn_model == None:
                        init_sv_cn()
                if inp_refs:
                    for path in inp_refs:
                        try:
                            refer, audio_tensor = get_spepc_for_gpu(hps, path, dtype, target_gpu, is_v2pro)
                            refers.append(refer)
                            if is_v2pro:
                                sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                        except Exception as e:
                            logger.error(e)
                if len(refers) == 0:
                    refers, audio_tensor = get_spepc_for_gpu(hps, refer_wav_path, dtype, target_gpu, is_v2pro)
                    refers = [refers]
                    if is_v2pro:
                        sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
            else:
                refer, audio_tensor = get_spepc_for_gpu(hps, refer_wav_path, dtype, target_gpu)
        
        # 8. 文本处理（与原始函数相同，但指定目标GPU）
        texts = text_segment.split("\n")
        audio_opt = []  # 存储所有音频片段
        
        for text in texts:
            if only_punc(text):
                continue
            
            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
            
            # 获取音素和BERT特征
            phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
            phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
            
            # 将BERT特征转移到目标GPU
            bert = torch.cat([bert1, bert2], 1)
            bert = bert.to(target_gpu).unsqueeze(0)
            
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(target_gpu).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(target_gpu)
            
            # 9. GPT推理（与原始函数相同，但指定目标GPU）
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            
            # 10. SoVITS解码（根据模型版本不同）
            if version not in {"v3", "v4"}:
                # v1/v2/v2Pro 版本
                if is_v2pro:
                    audio = (
                        vq_model.decode(
                            pred_semantic,
                            torch.LongTensor(phones2).to(target_gpu).unsqueeze(0),
                            refers,
                            speed=speed,
                            sv_emb=sv_emb,
                        )
                        .detach()
                        .cpu()
                        .numpy()[0, 0]
                    )
                else:
                    audio = (
                        vq_model.decode(
                            pred_semantic, 
                            torch.LongTensor(phones2).to(target_gpu).unsqueeze(0), 
                            refers, 
                            speed=speed
                        )
                        .detach()
                        .cpu()
                        .numpy()[0, 0]
                    )
            else:
                # v3/v4 版本
                phoneme_ids0 = torch.LongTensor(phones1).to(target_gpu).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(target_gpu).unsqueeze(0)
                
                fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                
                # 加载参考音频用于mel频谱
                ref_audio, sr = torchaudio.load(refer_wav_path)
                ref_audio = ref_audio.to(target_gpu).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                
                tgt_sr = 24000 if version == "v3" else 32000
                if sr != tgt_sr:
                    ref_audio = resample(ref_audio, sr, tgt_sr, target_gpu)
                
                mel2 = mel_fn(ref_audio) if version == "v3" else mel_fn_v4(ref_audio)
                mel2 = norm_spec(mel2)
                
                # ============ v3/v4 分块解码过程 ============
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                Tref = 468 if version == "v3" else 500
                Tchunk = 934 if version == "v3" else 1000
                if T_min > Tref:
                    mel2 = mel2[:, :, -Tref:]
                    fea_ref = fea_ref[:, :, -Tref:]
                    T_min = Tref

                chunk_len = Tchunk - T_min
                mel2 = mel2.to(dtype)
                fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)

                # 分块解码
                cfm_resss = []
                idx = 0
                while 1:
                    fea_todo_chunk = fea_todo[:, :, idx: idx + chunk_len]
                    if fea_todo_chunk.shape[-1] == 0:
                        break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    cfm_res = vq_model.cfm.inference(
                        fea, torch.LongTensor([fea.size(1)]).to(target_gpu), mel2, sample_steps, inference_cfg_rate=0
                    )
                    cfm_res = cfm_res[:, :, mel2.shape[2]:]
                    mel2 = cfm_res[:, :, -T_min:]
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)

                cfm_res = torch.cat(cfm_resss, 2)
                cfm_res = denorm_spec(cfm_res)

                # 根据版本选择声码器
                if version == "v3":
                    # 确保 bigvgan_model 在目标 GPU 上
                    if bigvgan_model is None:
                        init_bigvgan(target_gpu)
                    else:
                        try:
                            current_device = next(bigvgan_model.parameters()).device
                            if str(current_device) != target_gpu:
                                bigvgan_model = bigvgan_model.to(target_gpu)
                        except StopIteration:
                            bigvgan_model = bigvgan_model.to(target_gpu)
                    vocoder_model = bigvgan_model
                else:  # v4
                    # 确保 hifigan_model 在目标 GPU 上
                    if hifigan_model is None:
                        init_hifigan(target_gpu)
                    else:
                        try:
                            current_device = next(hifigan_model.parameters()).device
                            if str(current_device) != target_gpu:
                                hifigan_model = hifigan_model.to(target_gpu)
                        except StopIteration:
                            hifigan_model = hifigan_model.to(target_gpu)
                    vocoder_model = hifigan_model

                # 生成音频
                with torch.inference_mode():
                    # 确保输入数据类型与声码器一致
                    model_dtype = next(vocoder_model.parameters()).dtype
                    if cfm_res.dtype != model_dtype:
                        cfm_res = cfm_res.to(model_dtype)
                    
                    wav_gen = vocoder_model(cfm_res)
                    audio = wav_gen[0][0].cpu().detach().numpy()
            
            # 11. 音频归一化（每行独立归一化，防止削波）
            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            
            # 添加到音频列表
            audio_opt.append(audio)
            audio_opt.append(zero_wav)  # 添加静音段
        
        # 12. 合并所有音频片段
        if audio_opt:
            combined_audio = np.concatenate(audio_opt, axis=0)
            
            # 最终归一化
            max_audio = np.abs(combined_audio).max()
            if max_audio > 1.0:
                combined_audio = combined_audio / max_audio
            
            # 13. 确定采样率（根据模型版本）
            if version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
                sr = 32000
            elif version == "v3":
                sr = 48000 if if_sr else 24000
            else:  # v4
                sr = 48000
            
            # 14. 超分处理（仅v3且启用超分）
            if if_sr and version == "v3" and sr == 24000:
                audio_opt_tensor = torch.from_numpy(combined_audio).float().to(target_gpu)
                audio_opt_tensor, sr = audio_sr_for_gpu(audio_opt_tensor.unsqueeze(0), sr, target_gpu)
                combined_audio = audio_opt_tensor.cpu().numpy()[0]
                
                # 超分后再次归一化
                max_audio = np.abs(combined_audio).max()
                if max_audio > 1.0:
                    combined_audio = combined_audio / max_audio
                sr = 48000
            combined_audio = _validate_audio_output(combined_audio, sr)
            return combined_audio, sr
        else:
            # 如果没有生成音频，返回静音
            silence = np.zeros(16000, dtype=np.float32)
            return silence, 16000
            
    except Exception as e:
        logger.error(f"❌ GPU{gpu_index} 处理失败: {e}")
        raise

def get_spepc_for_gpu(hps, filename, dtype, target_gpu, is_v2pro=False):
    """
    为指定GPU获取频谱（修改自原get_spepc函数）
    """
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    
    # 确保音频在目标GPU上
    audio = audio.to(target_gpu)
    
    if sr0 != sr1:
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, target_gpu)
    else:
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
    
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, target_gpu).to(dtype)
    
    return spec, audio


def audio_sr_for_gpu(audio, sr, target_gpu):
    """
    为指定GPU进行音频超分
    """
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE
        try:
            sr_model = AP_BWE(target_gpu, DictToAttrRecursive)
        except FileNotFoundError:
            logger.info("你没有下载超分模型的参数，因此不进行超分。")
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)

def _get_model_specific_params(version, if_sr):
    """
    根据模型版本获取特定参数
    返回: (sample_rate, needs_vocoder, vocoder_type)
    """
    if version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        sr = 32000
        needs_vocoder = False
        vocoder_type = None
    elif version == "v3":
        if if_sr:
            sr = 48000  # 超分后
        else:
            sr = 24000  # 原始
        needs_vocoder = True
        vocoder_type = "bigvgan"
    else:  # v4
        sr = 48000
        needs_vocoder = True
        vocoder_type = "hifigan"
    
    return sr, needs_vocoder, vocoder_type

def _get_ssl_content(wav16k, target_gpu):
    """
    获取SSL内容，处理设备间数据传输
    """
    ssl_device = None
    try:
        if hasattr(ssl_model, 'parameters'):
            ssl_device = next(ssl_model.parameters()).device
        elif hasattr(ssl_model, 'device'):
            ssl_device = ssl_model.device
    except Exception as e:
        logger.warning(f"无法获取SSL模型设备，默认使用CPU: {e}")
        ssl_device = torch.device("cpu")
    
    # 根据SSL模型设备处理
    ssl_content = _get_ssl_content(wav16k, target_gpu)
    
    return ssl_content

def check_gpu_memory(gpu_index=0):
    """
    检查GPU内存使用情况
    """
    if not torch.cuda.is_available():
        return 0, 0
    
    try:
        torch.cuda.set_device(gpu_index)
        allocated = torch.cuda.memory_allocated(gpu_index) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(gpu_index) / 1024**3   # GB
        total = torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3
        
        free = total - allocated
        
        logger.debug(f"GPU{gpu_index}: 已用 {allocated:.2f}GB / 总计 {total:.2f}GB, 空闲 {free:.2f}GB")
        return free, total
    except Exception as e:
        logger.warning(f"检查GPU{gpu_index}内存失败: {e}")
        return 0, 0

def _resample_audio_if_needed(audio_data, orig_sr, target_sr):
    """
    如果需要，对音频进行重采样
    """
    if orig_sr == target_sr:
        return audio_data
    
    try:
        # 使用librosa进行高质量重采样
        resampled = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        logger.debug(f"音频重采样: {orig_sr}Hz -> {target_sr}Hz")
        return resampled
    except Exception as e:
        logger.error(f"音频重采样失败: {e}")
        # 如果重采样失败，返回原始音频并记录警告
        return audio_data

def _safe_model_load(model_func, *args, **kwargs):
    """
    安全的模型加载，带重试机制
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return model_func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"内存不足，尝试 {attempt+1}/{max_retries}...")
            torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # 等待1秒再重试
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

def _validate_audio_output(audio_data, sample_rate):
    """
    验证音频输出是否有效
    """
    if audio_data is None or len(audio_data) == 0:
        raise ValueError("生成的音频数据为空")
    
    if sample_rate <= 0:
        raise ValueError(f"无效的采样率: {sample_rate}")
    
    # 检查音频数据是否包含无效值
    if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
        logger.warning("音频数据包含NaN或Inf值，尝试修复...")
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 检查幅度是否过大
    max_amplitude = np.abs(audio_data).max()
    if max_amplitude > 10.0:  # 明显过大
        logger.warning(f"音频幅度过大 ({max_amplitude:.2f})，进行归一化")
        audio_data = audio_data / max_amplitude
    
    return audio_data
    
def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def handle_change(path, text, language):
    if is_empty(path, text, language):
        return JSONResponse(
            {"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400
        )

    if path != "" or path is not None:
        default_refer.path = path
    if text != "" or text is not None:
        default_refer.text = text
    if language != "" or language is not None:
        default_refer.language = language

    logger.info(f"当前默认参考音频路径: {default_refer.path}")
    logger.info(f"当前默认参考音频文本: {default_refer.text}")
    logger.info(f"当前默认参考音频语种: {default_refer.language}")
    logger.info(f"is_ready: {default_refer.is_ready()}")

    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def unload_model(speaker_id):
    if speaker_id in speaker_list and speaker_list[speaker_id].gpt is not None:
        speaker_list[speaker_id].gpt = None
        speaker_list[speaker_id].sovits = None
        torch.cuda.empty_cache()


def get_speaker_gpt_model(speaker_id, gpu_index=0):
    """获取指定说话人在指定GPU上的GPT模型"""
    speaker = speaker_list[speaker_id]
    if gpu_index == 0:
        return speaker.gpu0_gpt or speaker.gpt  # 回退到兼容字段
    elif gpu_index == 1:
        return speaker.gpu1_gpt
    else:
        raise ValueError(f"Invalid GPU index: {gpu_index}")

def get_speaker_sovits_model(speaker_id, gpu_index=0):
    """获取指定说话人在指定GPU上的Sovits模型"""
    speaker = speaker_list[speaker_id]
    if gpu_index == 0:
        return speaker.gpu0_sovits or speaker.sovits  # 回退到兼容字段
    elif gpu_index == 1:
        return speaker.gpu1_sovits
    else:
        raise ValueError(f"Invalid GPU index: {gpu_index}")
    
def split_long_text(text, threshold=long_text_threshold):
    """
    智能拆分长文本，尽量在自然停顿处拆分
    返回：(is_long, segments)
    - is_long: 是否为长文本
    - segments: 文本片段列表，长文本时为2段，短文本时为1段
    """
    # 单GPU环境不进行并行拆分
    if not IS_MULTI_GPU:
        return False, [text]
        
    # 多GPU环境拆分 
    if len(text) <= threshold:
        return False, [text]
    
    # 寻找最佳的拆分点（在句号、问号、感叹号、逗号等位置）
    split_positions = []
    for i in range(len(text)):
        if i > threshold * 0.3 and i < len(text) - threshold * 0.3:
            if text[i] in '。！？.!?；;，,':
                split_positions.append(i)
    
    # 如果没有找到合适的标点，在阈值位置强制拆分
    if not split_positions:
        split_pos = min(threshold, len(text) - 1)
    else:
        # 选择最接近中间位置的标点
        mid_point = len(text) // 2
        split_pos = min(split_positions, key=lambda x: abs(x - mid_point))
    
    # 确保拆分点不是最后一个字符
    split_pos = min(split_pos, len(text) - 5)
    
    return True, [text[:split_pos+1], text[split_pos+1:]]

# 6. 修改 handle 函数，调整 prompt 字段访问
def handle(
    refer_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    cut_punc,
    top_k,
    top_p,
    temperature,
    speed,
    inp_refs,
    sample_steps,
    if_sr,
    speaker_id="default"
):
    if speaker_id not in speaker_list:
        return JSONResponse({"code": 400, "message": f"speaker_id: {speaker_id} not found"}, status_code=400)

    # 使用 speaker_list 中定义的默认值
    if (
        refer_wav_path == "" or refer_wav_path is None
        or prompt_text == "" or prompt_text is None
        or prompt_language == "" or prompt_language is None
    ):
        refer_wav_path = speaker_list[speaker_id].prompt["ref_audio"] if refer_wav_path in ["", None] else refer_wav_path
        prompt_text = speaker_list[speaker_id].prompt["prompt_text"] if prompt_text in ["", None] else prompt_text
        prompt_language = speaker_list[speaker_id].prompt["prompt_lang"] if prompt_language in ["", None] else prompt_language

        # 如果仍然缺少必要参数，尝试使用全局默认参考音频
        if not is_full(refer_wav_path, prompt_text, prompt_language):
            refer_wav_path = default_refer.path if refer_wav_path in ["", None] else refer_wav_path
            prompt_text = default_refer.text if prompt_text in ["", None] else prompt_text
            prompt_language = default_refer.language if prompt_language in ["", None] else prompt_language
            if not default_refer.is_ready():
                return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    if sample_steps not in [4, 8, 16, 32]:
        sample_steps = 32

    if cut_punc is None:
        text = cut_text(text, default_cut_punc)
    else:
        text = cut_text(text, cut_punc)

    # 验证 prompt_language 和 text_language
    prompt_language = dict_language.get(prompt_language.lower(), prompt_language)
    text_language = dict_language.get(text_language.lower(), text_language)
    supported_languages = ["all_zh", "all_yue", "en", "all_ja", "all_ko", "zh", "yue", "ja", "ko", "auto", "auto_yue"]
    if prompt_language not in supported_languages:
        return JSONResponse({"code": 400, "message": f"prompt_language: {prompt_language} is not supported"}, status_code=400)
    if text_language not in supported_languages:
        return JSONResponse({"code": 400, "message": f"text_language: {text_language} is not supported"}, status_code=400)

    return StreamingResponse(
        get_tts_wav(
            refer_wav_path,
            prompt_text,
            prompt_language,
            text,
            text_language,
            top_k,
            top_p,
            temperature,
            speed,
            inp_refs,
            sample_steps,
            if_sr,
            spk=speaker_id
        ),
        media_type="audio/" + media_type,
    )


# --------------------------------
# 初始化部分
# --------------------------------

dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",  # 多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

# logger
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger("uvicorn")

# 获取配置
g_config = global_config.Config()

# 获取参数
parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument(
    "-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度"
)
parser.add_argument(
    "-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度"
)
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False
parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="流式返回模式, close / normal / keepalive")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式, wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="音频数据类型, int16 / int32")
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号设定, 符号范围,.;?!、，。？！；：…")
# 切割常用分句符为 `python ./api.py -cp ".?!。？！"`
parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")
parser.add_argument("-mm", "--max_models", type=int, default=3, help="最大同时加载模型数量")
parser.add_argument("-ltt", "--long_text_threshold", type=int, default=70, help="长文本界限")
parser.add_argument("--sd", "--subdomain", type=str, default=None, help="指定隧道使用的固定子域名 (例如: your-name)")
# 添加是否创建公开链接的参数
parser.add_argument("--public", action="store_true", default=False,help="是否创建公开链接 (默认不创建)")

args = parser.parse_args()
sovits_path = args.sovits_path
gpt_path = args.gpt_path
device = args.device
port = args.port
host = args.bind_addr
cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc
max_models = args.max_models
long_text_threshold = args.long_text_threshold

# 应用参数配置
default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

# 模型路径检查
if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    logger.warning(f"未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    logger.warning(f"未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    logger.info("未指定默认参考音频")
else:
    logger.info(f"默认参考音频路径: {default_refer.path}")
    logger.info(f"默认参考音频文本: {default_refer.text}")
    logger.info(f"默认参考音频语种: {default_refer.language}")

# 获取半精度
is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback
logger.info(f"半精: {is_half}")

# 流式返回模式
if args.stream_mode.lower() in ["normal", "n"]:
    stream_mode = "normal"
    logger.info("流式返回已开启")
else:
    stream_mode = "close"

# 音频编码格式
if args.media_type.lower() in ["aac", "ogg"]:
    media_type = args.media_type.lower()
elif stream_mode == "close":
    media_type = "wav"
else:
    media_type = "ogg"
logger.info(f"编码格式: {media_type}")

# 音频数据类型
if args.sub_type.lower() == "int32":
    is_int32 = True
    logger.info("数据类型: int32")
else:
    is_int32 = False
    logger.info("数据类型: int16")

# 初始化模型
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
ssl_model = cnhubert.get_model()
if is_half:
    bert_model = bert_model.half().to(device)
    ssl_model = ssl_model.half().to(device)
else:
    bert_model = bert_model.to(device)
    ssl_model = ssl_model.to(device)
change_gpt_sovits_weights(gpt_path=gpt_path, sovits_path=sovits_path)


# 1. 修改 speaker_list 初始化，添加默认 ref_audio、prompt_text 和 prompt_lang
#n_speaker#S
speaker_list = {}
#n_speaker#E

# --------------------------------
# 接口部分
# --------------------------------
app = FastAPI()

# 在接口部分添加 /voice/speakers 接口
@app.get("/voice/speakers")
async def get_speakers():
    # 只返回逻辑说话人列表，不暴露内部的双GPU结构
    speakers = [{"name": speaker_id} for speaker_id in speaker_list.keys()]
    return JSONResponse({"GPT-SOVITS": speakers}, status_code=200)

# 6. 修改 set_model 接口，支持 speaker_id
@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    return change_gpt_sovits_weights(
        gpt_path=json_post_raw.get("gpt_model_path"),
        sovits_path=json_post_raw.get("sovits_model_path"),
        speaker_id=json_post_raw.get("speaker_id", "default")
    )

@app.get("/set_model")
async def set_model(
    gpt_model_path: str = None,
    sovits_model_path: str = None,
    speaker_id: str = "default"
):
    return change_gpt_sovits_weights(gpt_path=gpt_model_path, sovits_path=sovits_model_path, speaker_id=speaker_id)

@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None):
    return handle_control(command)


@app.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"), json_post_raw.get("prompt_text"), json_post_raw.get("prompt_language")
    )


@app.get("/change_refer")
async def change_refer(refer_wav_path: str = None, prompt_text: str = None, prompt_language: str = None):
    return handle_change(refer_wav_path, prompt_text, prompt_language)


# 4. 修改 tts_endpoint POST 接口，支持 speaker_id
@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("cut_punc"),
        json_post_raw.get("top_k", 15),
        json_post_raw.get("top_p", 1.0),
        json_post_raw.get("temperature", 1.0),
        json_post_raw.get("speed", 1.0),
        json_post_raw.get("inp_refs", []),
        json_post_raw.get("sample_steps", 32),
        json_post_raw.get("if_sr", False),
        json_post_raw.get("speaker_id", "default")
    )


# 3. 修改 tts_endpoint GET 接口，添加 speaker_id 参数
@app.get("/")
async def tts_endpoint(
    refer_wav_path: str = None,
    prompt_text: str = None,
    prompt_language: str = None,
    text: str = None,
    text_language: str = None,
    cut_punc: str = None,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    speed: float = 1.0,
    inp_refs: list = Query(default=[]),
    sample_steps: int = 32,
    if_sr: bool = False,
    speaker_id: str = "default"  # 新增 speaker_id 参数
):
    return handle(
        refer_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        cut_punc,
        top_k,
        top_p,
        temperature,
        speed,
        inp_refs,
        sample_steps,
        if_sr,
        speaker_id
    )


if __name__ == "__main__":
    import threading
    import time
    import secrets
    
    # 1. 启动 FastAPI 服务器线程
    def run_server():
        uvicorn.run(app, host=host, port=port, workers=1)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"🚀 启动内部 FastAPI 服务器 (端口: {port})...")
    time.sleep(3)  # 等待服务器启动
    
    # 根据参数决定是否创建公开链接
    if args.public:
        print("🌐 正在创建公开隧道链接...")
        print("="*60)
        
        try:
            # 使用指定子域名或生成随机令牌
            share_token = args.sd if args.sd else secrets.token_urlsafe(32)
            
            public_url = setup_tunnel(
                local_host="127.0.0.1",
                local_port=port,
                share_token=share_token,
                share_server_address=None,
            )
            
            print(f"\n✅ 隧道创建成功！您的公开访问信息：")
            print("="*60)
            print(f"📢 公开 URL: {public_url}")
            print(f"🔧 API 根路径: {public_url}/")
            print(f"🎤 语音合成: {public_url}/voice")
            print(f"👥 说话人列表: {public_url}/voice/speakers")
            print(f"📊 模型信息: {public_url}/models/info")
            print(f"🖥️  系统状态: {public_url}/status")
            print("="*60)
            print("💡 提示：")
            print("   • 可通过GET或POST请求访问语音合成API")
            print("   • 此链接默认有效期为72小时")
            if args.sd:
                print(f"   • 使用固定子域名: {args.sd}")
            else:
                print(f"   • 使用随机令牌: {share_token[:16]}...")
            print("   • 按 Ctrl+C 可终止服务")
            print("="*60 + "\n")
            
        except requests.exceptions.ConnectionError:
            print(f"\n⚠️  网络错误：无法连接到 Gradio 隧道服务器")
            print(f"   这可能是因为网络限制")
            print(f"   ⚠️  公开链接创建失败，仅限本地访问")
            print(f"   本地访问地址：")
            print(f"   • http://localhost:{config.server_config.port}")
            print(f"   • http://0.0.0.0:{config.server_config.port}")
        except Exception as e:
            print(f"\n❌ 创建隧道时发生错误：{type(e).__name__}: {e}")
            print(f"   ⚠️  公开链接创建失败，仅限本地访问")
            print(f"   本地访问地址：http://localhost:{config.server_config.port}")
    else:
        # 不创建公开链接，仅显示本地地址
        print("🔒 未启用公开链接功能（如需启用请添加 --public 参数）")
        print("="*60)
        print(f"📢 本地访问地址：")
        print(f"   • http://localhost:{config.server_config.port}")
        print(f"   • http://0.0.0.0:{config.server_config.port}")
        if args.sd:
            print(f"   • 检测到子域名参数 --sd {args.sd}，但未启用公开链接")
        print("   • 按 Ctrl+C 可终止服务")
        print("="*60 + "\n")
    
    # 主循环保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 接收到中断信号，正在关闭服务...")


