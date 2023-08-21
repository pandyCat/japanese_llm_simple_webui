#!/bin/env python3

import argparse
import gradio as gr
import torch
import time
import numpy as np
from torch.nn import functional as F
import os
import re
from threading import Thread
#from peft import PeftModel
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


#------
# 設定
#------

# バージョン
VERSION = "1.0.0"

# ページの最上部に表示させたいタイトルを設定
TITLE_STRINGS = "LLM Cyber Deck"

# モデルタイプ("rinna","rinna4b","opencalm","llama","ja-stablelm","stablelm","bloom","falcon","mpt", "stablecode", "line", "weblab10b")
# ALL_MODEL_TYPE は ',' でsplitしてlistにする
ALL_MODEL_TYPES = "stablecode,ja-stablelm,rinna"

# ベースモデルを設定
ALL_BASE_MODELS = "stabilityai/japanese-stablelm-instruct-alpha-7b,stabilityai/japanese-stablelm-base-alpha-7b,rinna/japanese-gpt-neox-3.6b-instruction-ppo"
DICT_BASE_MODELS = {}
DICT_INSTANCE_MODELS = {}

# トークナイザ―の設定
ALL_TOKENIZER_MODELS = ["novelai/nerdstash-tokenizer-v1,novelai/nerdstash-tokenizer-v1,rinna/japanese-gpt-neox-3.6b-instruction-ppo,"]
DICT_TOKENIZER_MODEL = {}
DICT_INSTANCE_TOKENIZERS = {}

# モデルを8ビット量子化で実行するか("on","off")
LOAD_IN_8BIT = "off"
# モデルを4ビット量子化で実行するか("on","off") bitsandbytes 0.39.0 以降が必要
LOAD_IN_4BIT = "off"

# LoRAのディレクトリ(空文字列に設定すると読み込まない)
LORA_WEIGHTS = ""

# プロンプトタイプ("rinna","vicuna","alpaca","llama2","beluga","ja-stablelm","stablelm","redpajama","falcon","qa","none", "stablecode", "line", "weblab10b")
ALL_PROMPT_TYPES = "stablecode,ja-stablelm,rinna"

# model_type と model_prompt
DICT_MODEL_TO_PROMPT = {}

# プロンプトが何トークンを超えたら履歴を削除するか
PROMPT_THRESHOLD = 1024
# 履歴を削除する場合、何トークン未満まで削除するか
PROMPT_DELETED = 512

# 繰り返しペナルティ(大きいほど同じ繰り返しを生成しにくくなる)
REPETITION_PENALTY = 1.1
# 推論時に生成する最大トークン数
MAX_NEW_TOKENS = 512
# 推論時の出力の多様さ(大きいほどバリエーションが多様になる)
TEMPERATURE = 1.0

# WebUIがバインドするIPアドレス
GRADIO_HOST = "0.0.0.0" #'127.0.0.1'
# WebUIがバインドするポート番号
GRADIO_PORT = 7860

# WebUI上に詳細設定を表示するか
SETTING_VISIBLE = "on"

# デバッグメッセージを標準出力に表示するか("on","off")
DEBUG_FLAG = "on"

new_line = ""

#------------------
# クラス、関数定義
#------------------

class StopOnTokens(StoppingCriteria):
    def __init__(self, model_name:str) -> None:
        super().__init__()
        self.model_name = model_name

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # モデルからこのトークンIDが出力されたら生成をストップする
        if self.model_name == "llama":
            # 13="\n" (改行が出力されたらストップしたい場合は「13」も追加する)
            stop_ids = [2, 1 ,0]
        elif self.model_name == "stablelm" or self.model_name == "ja-stablelm" or self.model_name == 'stablecode':
            # 50278="<|USER|>"、50279="<|ASSISTANT|>"、50277="<|SYSTEM|>"、1="<|padding|>"、0="<|endoftext|>"
            stop_ids = [50278, 50279, 50277, 1, 0]
        elif self.model_name == "mpt":
            # 1="<|padding|>"、0="<|endoftext|>" (改行が出力されたらストップしたい場合は「187」も追加する)
            stop_ids = [1, 0]
        elif self.model_name == "falcon":
            # 193="\n"、11="<|endoftext|>" (改行が出力されたらストップしたい場合は「193」も追加する)
            stop_ids = [11]
        elif self.model_name == "xgen":
            stop_ids = [50256]
        elif self.model_name == "line" or self.model_name == "weblab10b":
            # ほとんどのトークナイザーは 1="<|padding|>"、0="<|endoftext|>"
            stop_ids = [1, 0]
        else:
            # ほとんどのトークナイザーは 1="<|padding|>"、0="<|endoftext|>"
            stop_ids = [1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def user(message, history, model_name:str):
    # Rinnaモデルの場合"<NL>"を改行に変換
    if model_name == "rinna":
        for item in history:
            item[0] = re.sub("<NL>", "\n", item[0])
            item[1] = re.sub("<NL>", "\n", item[1])
    # <br>が増殖するのを防止
    for item in history:
        item[0] = re.sub("<br>\n", "\n", item[0])
        item[1] = re.sub("<br>\n", "\n", item[1])
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

# Regenerateボタンクリック時の動作
def regen(history, model_name:str):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history[-1][1]=""
        # Rinnaモデルの場合"<NL>"を改行に変換
        if model_name == "rinna":
            for item in history:
                item[0] = re.sub("<NL>", "\n", item[0])
                item[1] = re.sub("<NL>", "\n", item[1])
        # <br>が増殖するのを防止
        for item in history:
            item[0] = re.sub("<br>\n", "\n", item[0])
            item[1] = re.sub("<br>\n", "\n", item[1])
        return history[-1][0], history

# Remove lastボタンクリック時の動作
def remove_last(history):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history.pop(-1)
        # <br>が増殖するのを防止
        for item in history:
            item[0] = re.sub("<br>\n", "\n", item[0])
            item[1] = re.sub("<br>\n", "\n", item[1])
        return history

# プロンプト文字列を生成する関数
def prompt(curr_system_message, history, prompt_name:str):

    # Rinna-3.6B形式のプロンプト生成
    if prompt_name == "rinna" or prompt_name == "rinna4b":
        messages = curr_system_message + \
            new_line.join([new_line.join(["ユーザー: "+item[0], "システム: "+item[1]])
                    for item in history])
    # Vicuna形式のプロンプト生成
    elif prompt_name == "vicuna":
        prefix = f"""A chat between a curious user and an artificial intelligence assistant.{new_line}The assistant gives helpful, detailed, and polite answers to the user's questions.{new_line}{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Alpaca形式のプロンプト生成
    elif prompt_name == "alpaca":
        prefix = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### Instruction:{new_line}"+item[0], f"{new_line}### Response:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Llama2 Chat形式のプロンプト生成
    elif prompt_name == "llama2":
        prefix = f"""System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join([f"User: "+item[0], f"Assistant: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # StableBeluga2形式のプロンプト生成
    elif prompt_name == "beluga":
        prefix = f"""### System:{new_line}You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### User:{new_line}"+item[0], f"{new_line}### Assistant:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages

    # stablecode-instruct形式のプロンプト生成
    elif prompt_name == "stablecode":

        # pndy
        # 改行があった場合、最初の行を user_query, 2行目以降を inputsとみなす
        messages_sub = ""
        prompts = history[0][0].split(new_line)
        for item in history:
            prompts = item[0].split(new_line)
            if len(prompts) > 1:
                inputs = f"{new_line}".join(prompts[1:])
            else:
                inputs = ""
            messages_sub += f"{new_line}".join([new_line.join([f"### Instruction\n"+prompts[0], f"### Inputs\n"+inputs, f"### Response: "+item[1]])])

        messages = curr_system_message + messages_sub

        prefix = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。{new_line}{new_line}"""
#        messages = curr_system_message + \
#            f"{new_line}".join([new_line.join([f"### 指示: "+item[0], f"### 応答: "+item[1]])
#                    for item in history])
        messages = prefix + messages

    # Japanese StableLM形式のプロンプト生成
    elif prompt_name == "ja-stablelm":

        # pndy
        # 改行があった場合、最初の行を user_query, 2行目以降を inputsとみなす
        messages_sub = ""
        prompts = history[0][0].split(new_line)
        for item in history:
            prompts = item[0].split(new_line)
            if len(prompts) > 1:
                inputs = f"{new_line}".join(prompts[1:])
            else:
                inputs = ""
            messages_sub += f"{new_line}".join([new_line.join([f"### 指示: "+prompts[0], f"### 入力"+inputs, f"### 応答: "+item[1]])])

        messages = curr_system_message + messages_sub

        prefix = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。{new_line}{new_line}"""
#        messages = curr_system_message + \
#            f"{new_line}".join([new_line.join([f"### 指示: "+item[0], f"### 応答: "+item[1]])
#                    for item in history])
        messages = prefix + messages

    # StableLM形式のプロンプト生成
    elif prompt_name == "stablelm":
        prefix = f"""<|SYSTEM|># StableLM Tuned (Alpha version){new_line}- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.{new_line}- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.{new_line}- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.{new_line}- StableLM will refuse to participate in anything that could harm a human.{new_line}"""
        messages = curr_system_message + \
            "".join(["".join([f"<|USER|>"+item[0], f"<|ASSISTANT|>"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Radpajama形式のプロンプト生成
    elif prompt_name == "redpajama":
        messages = curr_system_message + \
            new_line.join([new_line.join(["<human>: "+item[0], "<bot>: "+item[1]])
                    for item in history])
    # Falcon形式のプロンプト生成
    elif prompt_name == "falcon":
        messages = curr_system_message + \
            new_line.join([new_line.join(["User: "+item[0], "Asisstant:"+item[1]])
                    for item in history])
    # XGen形式のプロンプト生成
    elif prompt_name == "xgen":
        prefix = f"""A chat between a curious human and an artificial intelligence assistant.{new_line}The assistant gives helpful, detailed, and polite answers to the human's questions.{new_line}{new_line}"""
        messages = curr_system_message + \
                new_line.join([new_line.join(["### Human: "+item[0], "### Asisstant: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Q&A形式のプロンプト生成
    elif prompt_name == "qa":
        messages = curr_system_message + \
            new_line.join([new_line.join(["Q: "+item[0], "A: "+item[1]])
                    for item in history])
    # line形式のプロンプト生成
    elif prompt_name == "line":
        messages = curr_system_message + \
            new_line.join([new_line.join(["ユーザー: "+item[0], "システム: "+item[1]])
                    for item in history])
    elif prompt_name == "weblab10b":
        #text = f'以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{text}\n\n### 応答:'
        prefix = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}".join([new_line.join([f"### 指示: \n"+item[0], f"\n### 応答: "+item[1]])
                    for item in history])
        messages = prefix + messages

    # 特定の書式を使用しない(入力した文章の続きを生成する)場合のプロンプト生成
    elif prompt_name == "none" or prompt_name=="opencalm":
        messages = curr_system_message + \
            "".join(["".join([item[0], item[1]])
                    for item in history])
    # prompt_name設定が正しくなければ終了する
    else:
        print(f"Invalid prompt_name \"{prompt_name}\"")
        exit()
    # 生成したプロンプト文字列を返す
    return messages


def chat(curr_system_message, history, p_do_sample, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens, model_name):

    prompt_name = DICT_MODEL_TO_PROMPT[model_name]
    m = DICT_INSTANCE_MODELS[model_name]
    tok = DICT_INSTANCE_TOKENIZERS[model_name]

    # Initialize a StopOnTokens object
    stop = StopOnTokens(model_name=model_name)

    # "<br>"を削除しておく(モデルに付加された<br>タグが渡らないようにする)
    for item in history:
        item[0] = re.sub("<br>\n", "\n", item[0])
        item[1] = re.sub("<br>\n", "\n", item[1])

    # 会話履歴を表示
    if DEBUG_FLAG:
        print(f"history={history}\n")

    # プロンプト文字列生成
    del_flag = 0
    while True:
        # プロンプト文字列を生成する
        messages = prompt(curr_system_message, history, prompt_name=prompt_name)
        # プロンプトをトークナイザで変換する
        if model_name == "rinna":
            messages = re.sub("\n", "<NL>", messages)
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False, padding=True)
        elif model_name == "rinna4b":
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False)
        elif model_name == "opencalm":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "llama":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "stablecode":
            model_inputs = tok([messages], return_tensors="pt", return_token_type_ids=False) #add_special_tokens=False)
        elif model_name == "ja-stablelm":
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False)
        elif model_name == "stablelm":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "bloom":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "falcon":
            model_inputs = tok([messages], return_tensors="pt")
            model_inputs.pop('token_type_ids')
        elif model_name == "mpt":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "xgen":
            model_inputs = tok([messages], return_tensors="pt")
        elif model_name == "line":
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False)
        elif model_name == "weblab10b":
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False)
            model_inputs.pop('token_type_ids')
        # もしプロンプトのトークン数が多すぎる場合は削除フラグを設定
        if del_flag == 0 and len(model_inputs['input_ids'][0]) > PROMPT_THRESHOLD:
            del_flag = 1
        # 削除フラグが設定され、かつPROMPT_DELETEDよりトークン数が多い場合は履歴の先頭を削除
        if del_flag == 1 and len(model_inputs['input_ids'][0]) > PROMPT_DELETED:
            history.pop(0)
        # 削除フラグが設定されてないか、設定されているがPROMPT_DELETEDよりトークン数が少ない場合ループを抜ける
        else:
            break

    # プロンプトを標準出力に表示
    if DEBUG_FLAG:
        print(f"--prompt strings--\n{messages}\n----\n")
        print(f"--prompt tokens--\n{model_inputs}\n----\n")

    # 入力トークンをGPUに送る
    model_inputs = model_inputs.to("cuda")

    # モデルに入力して回答を生成(ストリーミング出力させる)
    streamer = TextIteratorStreamer(
        tok, timeout=60., skip_prompt=True, skip_special_tokens=True)

    if DEBUG_FLAG:
        print(f"do_sample={p_do_sample}")
        print(f"temperature={p_temperature}")
        print(f"top_k={p_top_k}")
        print(f"top_p={p_top_p}")
        print(f"repetition_penalty={p_repetition_penalty}")
        print(f"max_new_tokens={p_max_new_tokens}")

    # 推論設定
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=p_max_new_tokens,
        do_sample=p_do_sample,
        top_k=p_top_k,
        top_p=p_top_p,
        temperature=p_temperature,
        num_beams=1,
        repetition_penalty=p_repetition_penalty,
#        pad_token_id=tok.pad_token_id,
#        bos_token_id=tok.bos_token_id,
#        eos_token_id=tok.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    if tok.pad_token_id != None:
        generate_kwargs["pad_token_id"] = tok.pad_token_id
    if tok.bos_token_id != None:
        generate_kwargs["bos_token_id"] = tok.bos_token_id
    if tok.eos_token_id != None:
        generate_kwargs["eos_token_id"] = tok.eos_token_id
    
    if model_name == "line":
        generate_kwargs['pad_token_id']=tok.pad_token_id
        
    t = Thread(target=m.generate, kwargs=generate_kwargs)

    # スレッドで生成を実行
    t.start()

    #print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # Rinnaモデルの場合"<NL>"を改行に変換
        if model_name == "rinna":
            new_text = re.sub("<NL>", "\n", new_text)
        # XGenモデルの場合<|endoftext|>は表示させない
        if model_name == "xgen":
            new_text = re.sub("^<\|endoftext\|>$", "", new_text)
        #print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    if DEBUG_FLAG:
        print(f"--generated strings--\n{partial_text}\n----\n")
    return partial_text

def loadModelAndPromp(model_name: str, tokenizer_name: str):

    global new_line

    model_data_name     = DICT_BASE_MODELS[model_name]
    tokenizer_data_name = DICT_TOKENIZER_MODEL[model_name]
    
    # Rinna-3.6Bモデルの場合
    if model_name == "rinna":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 改行を示す文字の設定
        new_line = "<NL>"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name, 
            torch_dtype=torch.float16, 
            load_in_8bit=LOAD_IN_8BIT, 
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name, use_fast=False)
        print(f"Sucessfully loaded the tokenizer to the memory")
        # padding設定
        m.config.pad_token_id = tok.eos_token_id
    # Rinna-4Bモデルの場合
    elif model_name == "rinna4b":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name, 
            torch_dtype=torch.float16, 
            load_in_8bit=LOAD_IN_8BIT, 
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name, use_fast=False)
        print(f"Sucessfully loaded the tokenizer to the memory")
        # padding設定
        m.config.pad_token_id = tok.eos_token_id
    # Open CALMモデルの場合
    elif model_name == "opencalm":
        from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name, 
            torch_dtype=torch.float16, 
            load_in_8bit=LOAD_IN_8BIT, 
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = GPTNeoXTokenizerFast.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")
    # Llama系モデルの場合
    elif model_name == "llama":
        from transformers import LlamaForCausalLM, LlamaTokenizer
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = LlamaForCausalLM.from_pretrained(
            model_data_name, 
            torch_dtype=torch.float16, 
            load_in_8bit=LOAD_IN_8BIT, 
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto',
            rope_scaling={"type": "dynamic", "factor": 2.0}
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = LlamaTokenizer.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")

    # stablecode の場合
    elif  model_name == "stablecode":
        # 改行を示す文字の設定
        new_line = "\n"

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")

        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(model_data_name,
        trust_remote_code=True,
        torch_dtype="auto",
        )
        m.eval()
        assert torch.cuda.is_available()
        m = m.cuda()
        print(f"Sucessfully loaded the model to the memory")

    # Japanese StableLMモデルの場合
    elif model_name == "ja-stablelm":

        # 改行を示す文字の設定
        new_line = "\n"

        from transformers import LlamaTokenizer, AutoModelForCausalLM

        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
        print(f"Sucessfully loaded the tokenizer to the memory")

        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-alpha-7b",    
            trust_remote_code=True,
        )
        m.half()
        m.eval()
        assert torch.cuda.is_available()
        m = m.to("cuda")
        print(f"Sucessfully loaded the model to the memory")

    # StableLMモデルの場合
    elif model_name == "stablelm":
        from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = GPTNeoXTokenizerFast.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")
    # Bloomモデルの場合
    elif model_name == "bloom":
        from transformers import AutoModelForCausalLM, BloomTokenizerFast
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT, 
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = BloomTokenizerFast.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")
    # Falconモデルの場合
    elif model_name == "falcon":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT, 
            trust_remote_code=True,
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")
    # MPTモデルの場合
    elif model_name == "mpt":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT, 
            trust_remote_code=True,
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name)
        print(f"Sucessfully loaded the tokenizer to the memory")
    # Xgenモデルの場合
    elif model_name == "xgen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            load_in_8bit=LOAD_IN_8BIT,
            device_map='auto'
            )
        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name, trust_remote_code=True)
        print(f"Sucessfully loaded the tokenizer to the memory")
    elif model_name == "line":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            device_map="auto")

        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name, use_fast=False)
        print(f"Sucessfully loaded the tokenizer to the memory")
    elif model_name == "weblab10b":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # 改行を示す文字の設定
        new_line = "\n"
        # モデルのロード
        print(f"Starting to load the model \"{model_data_name}\" to memory")
        m = AutoModelForCausalLM.from_pretrained(
            model_data_name,
            torch_dtype=torch.float16,
            device_map="auto")

        print(f"Sucessfully loaded the model to the memory")
        # トークナイザ―のロード
        print(f"Starting to load the tokenizer \"{tokenizer_data_name}\" to memory")
        tok = AutoTokenizer.from_pretrained(tokenizer_data_name, use_fast=False)
        print(f"Sucessfully loaded the tokenizer to the memory")


    # name設定が正しくなければ終了する
    else:
        print(f"Invalid model_name \"{model_name}\"")
        exit()

    DICT_INSTANCE_MODELS[model_name] = m
    DICT_INSTANCE_TOKENIZERS[model_name] = tok  # モデルネームで引けるようにしよう

    return




#------
# 実行
#------

# 引数を取得
parser = argparse.ArgumentParser()
parser.add_argument("--all-base-models", type=str, default=ALL_BASE_MODELS, help="モデル名またはディレクトリのパス")
parser.add_argument("--all-model-types", type=str, default="ja-stablelm", help="モデルタイプ名")
#parser.add_argument("--all-model-types", type=str, choices=["rinna", "rinna4b", "opencalm", "llama", "ja-stablelm", "stablelm", "bloom", "falcon", "mpt", "xgen", "stablecode","line", "weblab10b"], default="ja-stablelm", help="モデルタイプ名")
parser.add_argument("--all-tokenizer-models", type=str, default=ALL_TOKENIZER_MODELS, help="トークナイザー名またはディレクトリのパス")
parser.add_argument("--load-in-8bit", type=str, choices=["on", "off"], default=LOAD_IN_8BIT, help="8bit量子化するかどうか")
parser.add_argument("--load-in-4bit", type=str, choices=["on", "off"], default=LOAD_IN_4BIT, help="4bit量子化するかどうか")
parser.add_argument("--lora", type=str, default=LORA_WEIGHTS, help="LoRAディレクトリのパス")
parser.add_argument("--all-prompt-types", type=str, default="ja-stablelm", help="プロンプトタイプ名")
#parser.add_argument("--all-prompt-types", type=str, choices=["rinna", "vicuna", "alpaca", "llama2", "beluga", "ja-stablelm", "stablelm", "redpajama", "falcon", "xgen", "qa", "none", "stablecode", "line", "weblab10b"], default="ja-stablelm", help="プロンプトタイプ名")
parser.add_argument("--prompt-threshold", type=int, default=PROMPT_THRESHOLD, help="このトークン数を超えたら古い履歴を削除")
parser.add_argument("--prompt-deleted", type=int, default=PROMPT_DELETED, help="古い履歴削除時にこのトークン以下にする")
parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY, help="繰り返しに対するペナルティ")
parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="推論時に生成するトークン数の最大")
parser.add_argument("--setting-visible", type=str, choices=["on", "off"], default=SETTING_VISIBLE, help="詳細設定を表示するかどうか")
parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="生成する文章の多様さ")
parser.add_argument("--host", type=str, default=GRADIO_HOST, help="WebサーバがバインドするIPアドレスorホスト名")
parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Webサーバがバインドするポート番号")
parser.add_argument("--title", type=str, default=TITLE_STRINGS, help="Webページのタイトル")
parser.add_argument("--debug", type=str, choices=["on", "off"], default=DEBUG_FLAG, help="デバッグメッセージを標準出力に表示")
args = parser.parse_args()

# 引数でセットされた値で上書きする
ALL_BASE_MODELS = args.all_base_models.split(',')
ALL_MODEL_TYPES = args.all_model_types.split(',')
ALL_TOKENIZER_MODELS = args.all_tokenizer_models.split(',')
LOAD_IN_8BIT = args.load_in_8bit
LOAD_IN_4BIT = args.load_in_4bit
LORA_WEIGHTS = args.lora
ALL_PROMPT_TYPES = args.all_prompt_types.split(',')
PROMPT_THRESHOLD = args.prompt_threshold
PROMPT_DELETED = args.prompt_deleted
REPETITION_PENALTY=args.repetition_penalty
MAX_NEW_TOKENS = args.max_new_tokens
SETTING_VISIBLE = args.setting_visible
TEMPERATURE = args.temperature
GRADIO_HOST = args.host
GRADIO_PORT = args.port
TITLE_STRINGS = args.title
DEBUG_FLAG = args.debug

# modelの名前とtokenizerの名前の辞書
assert len(ALL_MODEL_TYPES) == len(ALL_PROMPT_TYPES)
for cc in range(len(ALL_MODEL_TYPES)):
    DICT_MODEL_TO_PROMPT[ALL_MODEL_TYPES[cc]] = ALL_PROMPT_TYPES[cc]

assert len(ALL_MODEL_TYPES) == len(ALL_BASE_MODELS)
for cc in range(len(ALL_MODEL_TYPES)):
    DICT_BASE_MODELS[ALL_MODEL_TYPES[cc]] = ALL_BASE_MODELS[cc]

assert len(ALL_MODEL_TYPES) == len(ALL_TOKENIZER_MODELS)
for cc in range(len(ALL_MODEL_TYPES)):
    DICT_TOKENIZER_MODEL[ALL_MODEL_TYPES[cc]] = ALL_TOKENIZER_MODELS[cc]

# パラメータ表示
print("---- パラメータ ----")
print(f"モデル名orパス: {ALL_BASE_MODELS}")
print(f"モデルタイプ名: {ALL_MODEL_TYPES}")
print(f"トークナイザー: {ALL_TOKENIZER_MODELS}")
print(f"8bit量子化: {LOAD_IN_8BIT}")
print(f"4bit量子化: {LOAD_IN_4BIT}")
if LORA_WEIGHTS == "":
    print(f"LoRAモデルパス: (LoRAなし)")
else:
    print(f"LoRAモデルパス: {LORA_WEIGHTS}")
print(f"プロンプトタイプ: {ALL_PROMPT_TYPES}")
print(f"プロンプトトークン数しきい値: {PROMPT_THRESHOLD}")
print(f"プロンプトトークン数削除値: {PROMPT_DELETED}")
print(f"繰り返しペナルティ: {REPETITION_PENALTY}")
print(f"生成最大トークン数: {MAX_NEW_TOKENS}")
print(f"詳細設定表示: {SETTING_VISIBLE}")
print(f"Temperature: {TEMPERATURE}")
print(f"WebサーバIPorホスト名: {GRADIO_HOST}")
print(f"Webサーバポート番号: {GRADIO_PORT}")
print(f"Webページタイトル: {TITLE_STRINGS}")
print(f"デバッグ: {DEBUG_FLAG}\n")

# LOAD_IN_8BITはTrue or Falseに変換
if LOAD_IN_8BIT == "on":
    LOAD_IN_8BIT = True
else:
    LOAD_IN_8BIT = False
# LOAD_IN_4BITはTrue or Falseに変換
if LOAD_IN_4BIT == "on":
    LOAD_IN_4BIT = True
else:
    LOAD_IN_4BIT = False
# SETTING_VISIBLEはTrue or Falseに変換
if SETTING_VISIBLE == "on":
    SETTING_VISIBLE = True
else:
    SETTING_VISIBLE = False
# DEBUG_FLAGはTrue or Falseに変換
if DEBUG_FLAG == "on":
    DEBUG_FLAG = True
else:
    DEBUG_FLAG = False

## モデルタイプによる設定とモデルのロード
for model_name in ALL_MODEL_TYPES:
    tokenizer_name = DICT_MODEL_TO_PROMPT[model_name]
    loadModelAndPromp(model_name=model_name, tokenizer_name=tokenizer_name)

# ジェネレータの作成 (不要だと思われるためコメントアウト)
#generator = pipeline('text-generation', model=m, tokenizer=tok)

## LoRAのロード
#if LORA_WEIGHTS != "":
#    print(f"Starting to load the LoRA weights \"{LORA_WEIGHTS}\" to memory")
#    m = PeftModel.from_pretrained(m, LORA_WEIGHTS, torch_dtype=torch.float16)
#    print(f"Sucessfully loaded the LoRA weights to the memory")

# プロンプトの先頭に付加する文字列
start_message = ""


# Webページ
with gr.Blocks(title="LLM Simple WebUI", theme=gr.themes.Base()) as demo:
    history = gr.State([])
    gr.Markdown(f"## {TITLE_STRINGS}")
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        p_model_list = gr.Dropdown(ALL_MODEL_TYPES, label="どのLLMかえらんでね！",value=ALL_MODEL_TYPES[0])
    with gr.Row():
        with gr.Column(scale=20):
            msg = gr.Textbox(label="Chat Message Box", placeholder="ここに入力",
                             show_label=False).style(container=False)
        with gr.Column(scale=1, min_width=100):
            submit = gr.Button("Submit")
    with gr.Row():
                stop = gr.Button("Stop")
                regenerate = gr.Button("Regenerate")
                removelast = gr.Button("Remove last")
                clear = gr.Button("Clear")
    with gr.Accordion(label="Advanced Settings", open=False, visible=SETTING_VISIBLE):
        with gr.Blocks():
            p_do_sample = gr.Radio([True, False], value=True, label="Do Sample")
            p_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=TEMPERATURE, step=0.1, label="Temperature", interactive=True)
            p_top_k = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Top_K (0=無効)", interactive=True)
            p_top_p = gr.Slider(minimum=0.01, maximum=1.00, value=1.00, step=0.01, label="Top_P (1.00=無効)", interactive=True)
        with gr.Blocks():
            p_max_new_tokens = gr.Slider(minimum=1, maximum=2048, value=MAX_NEW_TOKENS, step=1, label="Max New Tokens", interactive=True)
            p_repetition_penalty = gr.Slider(minimum=1.00, maximum=5.00, value=REPETITION_PENALTY, step=0.01, label="Repetition Penalty (1.00=ペナルティなし)", interactive=True)
    with gr.Row():
        str_llm_enum = "\n".join(str(kk) + "\t\t: " + str(vv)
            for kk, vv in DICT_BASE_MODELS.items())
        gr.Text(str_llm_enum, label="ローカルで動作中のLLMインスタンスは以下の通りです")

    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot, p_model_list], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, p_do_sample, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens, p_model_list], outputs=[chatbot], queue=True)

    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot, p_model_list], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, p_do_sample, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens, p_model_list], outputs=[chatbot], queue=True)

    regenerate_click_event = regenerate.click(fn=regen, inputs=[chatbot, p_model_list], outputs=[msg, chatbot], queue=False).then(
               lambda: None, None, [msg], queue=False).then(
                   fn=chat, inputs=[system_msg, chatbot, p_do_sample, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens, p_model_list], outputs=[chatbot], queue=True)

    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event, regenerate_click_event], queue=False)

    removelast.click(fn=remove_last, inputs=[chatbot], outputs=[chatbot], queue=False)

    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=2)
demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)
