import os
os.environ["HF_HOME"]           = "/media/volume/yueru-s-data/hf_cache"
os.environ["TRANSFORMERS_CACHE"]= "/media/volume/yueru-s-data/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/media/volume/yueru-s-data/hf_cache"
os.environ["HF_METRICS_CACHE"]  = "/media/volume/yueru-s-data/hf_cache"

# now you can safely import everything else
from typing import List, Literal, TypedDict, Dict, Any
import torch
import shutil

torch._dynamo.disable()

import unsloth
from unsloth import FastLanguageModel
# … other imports …
from typing import List, Literal, TypedDict
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from cos.model_paths import *

TaskType = Literal["qualitative", "lambda_inference"]
SupportedModel = Literal["gptj", "t0pp", "llama-7b", "llama-7b-chat", "alpaca", "mistral"]

class Task(TypedDict):
    prompt: str
    true_context: str
    probes: List[str]
    lambdas: List[float]

HF_MODEL_CLASS = {
    "gptj": AutoModelForCausalLM,
    "t0pp": AutoModelForSeq2SeqLM,
    "llama-2-7b": AutoModelForCausalLM,
    "llama-2-7b-chat": AutoModelForCausalLM,
    "llama-2-13b": AutoModelForCausalLM,
    "llama-2-13b-chat": AutoModelForCausalLM,
    "llama-2-dummy": AutoModelForCausalLM,
    "llama-3-8b": AutoModelForCausalLM,
    "llama-3-8b-chat": AutoModelForCausalLM,
    "alpaca": AutoModelForCausalLM,
    "mistral": AutoModelForCausalLM
}

HF_MODEL_SOURCE = {
    "gptj": "EleutherAI/gpt-j-6B",
    "t0pp": "bigscience/T0pp",
    "llama-2-7b": HF_LLAMA_7B_TEXT_DIR,
    "llama-2-7b-chat": HF_LLAMA_7B_CHAT_DIR,
    "llama-2-13b": HF_LLAMA_13B_TEXT_DIR,
    "llama-2-13b-chat": HF_LLAMA_13B_CHAT_DIR,
    "llama-2-dummy": "mlabonne/dummy-llama-2",
    "llama-3-8b": HF_LLAMA_3_8B_TEXT_DIR,
    "llama-3-8b-chat": HF_LLAMA_3_8B_CHAT_DIR,
    "alpaca": HF_ALPACA_DIR,
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}

CACHE_ENV = "/media/volume/yueru-s-data/hf_cache"
# Ensure HF cache directories are set before any HF import
os.environ.setdefault("HF_HOME", CACHE_ENV)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_ENV)
os.environ.setdefault("HF_DATASETS_CACHE", CACHE_ENV)
os.environ.setdefault("HF_METRICS_CACHE", CACHE_ENV)

# Remove any stale compiled cache to force fresh load
compiled_cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
if os.path.exists(compiled_cache_dir):
    shutil.rmtree(compiled_cache_dir)


def load_hf_model_and_tokenizer(
    model_name: str,
    use_unsloth: bool = True,
    from_pretrained_kwargs: Dict[str, Any] = None
):
    """
    Loads a model and tokenizer. Pass overrides via `from_pretrained_kwargs`.

    Default Unsloth kwargs enforce no compilation and exact model name:
      - load_in_4bit: True
      - fast_inference: False  # disable torch.compile
      - use_exact_model_name: True
      - cache_dir: HF_HOME
    """
    from_pretrained_kwargs = from_pretrained_kwargs or {}

    if use_unsloth:
        default_args = {
            "model_name": model_name,
            "max_seq_length": 1024,
            "dtype": None,
            "load_in_4bit": True,
            "fast_inference": False,
            "use_exact_model_name": True,
            "cache_dir": os.environ["HF_HOME"],
        }
        # Merge user overrides last so they can override defaults
        hf_args = {**default_args, **from_pretrained_kwargs}
        model, tokenizer = FastLanguageModel.from_pretrained(**hf_args)
    else:
        model_class = HF_MODEL_CLASS.get(model_name)
        ckpt_dir = HF_MODEL_SOURCE.get(model_name)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if "llama-2" in model_name:
            model = model_class.from_pretrained(
                ckpt_dir, device_map='cuda', torch_dtype=torch.float16
            )
        elif "llama-3" in model_name:
            model = model_class.from_pretrained(
                ckpt_dir, device_map='cuda', torch_dtype=torch.bfloat16
            )
        else:
            model = model_class.from_pretrained(
                ckpt_dir, device_map='cuda'
            ).cuda()

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer
# Usage example:
# model, tokenizer = load_hf_model_and_tokenizer(
#     "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
#     from_pretrained_kwargs={"load_in_4bit": False, "use_exact_model_name": True}
# )


def assert_dialog(dialog):
    assert type(dialog) == list
    for i, msg in enumerate(dialog):
        assert type(msg) == dict
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] in ["user", "assistant", "system"]
        if i % 2 == 0:
            assert msg["role"] == "user"
        else:
            assert msg["role"] == "assistant"

def get_context_pair_dialogs(
    prompts: List[str], 
    contexts: List[str], 
    put_context_first: bool = False,
    is_history: bool = False
):
    """
    Returns dialogs with context and dialogs without context by batching prompts and contexts.

    Args:
        put_context_first(bool): specifies whether the context should be put first in the prompt,
    e.g. put_context_first=True would return "context prompt" instead of "prompt context" for each of
    the context dialogs.
    """
    def return_pair(prompt, context, is_history=is_history):
        # l2: prompt (l_1) + an chosen context
        if len(context) == 0:
            l_2 = prompt
            d = [{"role": "user", "content": l_2},] #? meaning of the ",”
        elif len(prompt) == 0:
            l_2 = context
            d = [{"role": "user", "content": l_2},]
        else:
            if is_history:
                print(f"Using history context: {context}")
                d_tmp = [{"role": "user", "content": prompt},]
                d = context + d_tmp
            else:
                l_2 = " ".join([context, prompt]) if put_context_first else " ".join([prompt, context])
                d = [{"role": "user", "content": l_2},]
        d_nc = [{"role": "user", "content": prompt}]
        return d, d_nc
    
    assert len(prompts) == len(contexts)
    dialogs, dialogs_nc = [], []
    for prompt, context in zip(prompts, contexts):
        cd, ncd = return_pair(prompt, context)
        dialogs.append(cd)
        dialogs_nc.append(ncd)
    return dialogs, dialogs_nc


def get_context_pair_texts(
    prompts: List[str], 
    contexts: List[str], 
    put_context_first: bool = False
):
    """
    Returns text with context and dialogs without context by batching prompts and contexts.

    Args:
        put_context_first(bool): specifies whether the context should be put first in the prompt,
    e.g. put_context_first=True would return "context prompt" instead of "prompt context" for each of
    the context dialogs.
    """
    def return_pair(prompt, context):
        if len(context) == 0:
            l_2 = prompt
        elif len(prompt) == 0:
            l_2 = context
        else:
            l_2 = " ".join([context, prompt]) if put_context_first else " ".join([prompt, context])
        return l_2, prompt
    
    assert len(prompts) == len(contexts)
    texts, texts_nc = [], []
    for prompt, context in zip(prompts, contexts):
        t_c, t_nc = return_pair(prompt, context)
        texts.append(t_c)
        texts_nc.append(t_nc)

    return texts, texts_nc


def get_multi_context_pair_dialogs(
    prompts: List[List[str]], 
    all_contexts: List[List[str]], 
    put_context_first: bool = False
):
    """
    Multi-directional context steering.

    Args:
        put_context_first(bool): specifies whether the context should be put first in the prompt,
    e.g. put_context_first=True would return "context prompt" instead of "prompt context" for each of
    the context dialogs.
    """
    def return_multi_pair(prompt, contexts):
        d = [
            [{"role": "user", "content": (
                prompt if not c else
                c if not prompt else
                f"{c} {prompt}" if put_context_first else f"{prompt} {c}"
            )}]
            for c in contexts
        ]
        return d, [{"role": "user", "content": prompt}]

    dialogs, dialogs_nc = [[] for _ in range(len(all_contexts))], []
    for i in range(len(prompts)):
        assert len(prompts) == len(all_contexts[i])
        prompt = prompts[i]
        contexts = [c[i] for c in all_contexts]
        cd, ncd = return_multi_pair(prompt, contexts)
        for j in range(len(cd)):
            dialogs[j].append(cd[j])
        dialogs_nc.append(ncd)
    return dialogs, dialogs_nc

def get_multi_context_pair_texts(
    prompts: List[str], 
    all_contexts: List[List[str]], 
    put_context_first: bool = False
):
    """
    Returns text with context and dialogs without context by batching prompts and contexts.

    Args:
        put_context_first(bool): specifies whether the context should be put first in the prompt,
    e.g. put_context_first=True would return "context prompt" instead of "prompt context" for each of
    the context dialogs.
    """
    def return_multi_pair(prompt, contexts):
        l_2s = [
            prompt if not c else
            c if not prompt else
            f"{c} {prompt}" if put_context_first else f"{prompt} {c}"
            for c in contexts
        ]
        return l_2s, prompt
    
    assert len(prompts) == len(all_contexts)
    texts, texts_nc = [[] for _ in range(len(all_contexts))], []
    for i in range(len(prompts)):
        prompt = prompts[i]
        contexts = [c[i] for c in all_contexts]
        t_c, t_nc = return_multi_pair(prompt, contexts)
        for j in range(len(t_c)):
            texts[j].append(t_c[j])
        texts_nc.append(t_nc)

    return texts, texts_nc


def tile_seqs(seqs, n):
    """
    Taking in seqs=[a, b], n=3, will output [a, a, a, b, b, b]
    """
    assert type(seqs) == list
    output = [seq for seq in seqs for _ in range(n)]

    # sanity check
    # assert [i == seqs[0] for i in output[:n]]

    return output

def repeat_seqs(seqs, n):
    """
    Taking in seqs=[a, b], n=3, will output [a, b, a, b, a, b]
    """
    assert type(seqs) == list
    output = seqs * n 

    # sanity check
    # assert output[:len(seqs)] == seqs

    return output



def sample_top_p(probs, p):
    """Taken from meta-llama/llama
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
