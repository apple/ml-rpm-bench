import json
import copy as cp
import os
import os.path as osp
import random as rd
import sys
import time
from abc import abstractmethod
from functools import partial
from PIL import Image
import google.generativeai as genai
import numpy as np
import requests
import shutil
from utils import *

APIBASES = {
    "OFFICIAL": "https://api.openai.com/v1/chat/completions",
}


class BaseAPI:

  def __init__(
      self,
      retry=10,
      wait=3,
      system_prompt=None,
      verbose=True,
      fail_msg="Failed to obtain answer via API.",
      **kwargs,
  ):
    self.wait = wait
    self.retry = retry
    self.system_prompt = system_prompt
    self.kwargs = kwargs
    self.verbose = verbose
    self.fail_msg = fail_msg
    self.logger = get_logger("ChatAPI")
    if len(kwargs):
      self.logger.info(f"BaseAPI received the following kwargs: {kwargs}")
      self.logger.info(f"Will try to use them as kwargs for `generate`. ")

  @abstractmethod
  def generate_inner(self, inputs, **kwargs):
    self.logger.warning(f"For APIBase, generate_inner is an abstract method. ")
    assert 0, "generate_inner not defined"
    ret_code, answer, log = None, None, None
    # if ret_code is 0, means succeed
    return ret_code, answer, log

  def generate(self, inputs, **kwargs):
    input_type = None
    if isinstance(inputs, str):
      input_type = "str"
    elif isinstance(inputs, list) and isinstance(inputs[0], str):
      input_type = "strlist"
    elif isinstance(inputs, list) and isinstance(inputs[0], dict):
      input_type = "dictlist"
    assert input_type is not None, input_type

    answer = None
    for i in range(self.retry):
      T = rd.random() * self.wait * 2
      time.sleep(T)
      try:
        ret_code, answer, log = self.generate_inner(inputs, **kwargs)
        if ret_code == 0 and self.fail_msg not in answer and answer != "":
          if self.verbose:
            print(answer)
          return answer
        elif self.verbose:
          self.logger.info(f"RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}")
      except Exception as err:
        if self.verbose:
          self.logger.error(f"An error occured during try {i}:")
          self.logger.error(err)

    return self.fail_msg if answer in ["", None] else answer


def GPT_context_window(model):
  length_map = {
      "gpt-4-1106-preview": 128000,
      "gpt-4-vision-preview": 128000,
      "gpt-4": 8192,
      "gpt-4-32k": 32768,
      "gpt-4-0613": 8192,
      "gpt-4-32k-0613": 32768,
      "gpt-3.5-turbo-1106": 16385,
      "gpt-3.5-turbo": 4096,
      "gpt-3.5-turbo-16k": 16385,
      "gpt-3.5-turbo-instruct": 4096,
      "gpt-3.5-turbo-0613": 4096,
      "gpt-3.5-turbo-16k-0613": 16385,
  }
  if model in length_map:
    return length_map[model]
  else:
    return 4096


class OpenAIWrapper(BaseAPI):
  is_api: bool = True

  def __init__(
      self,
      model: str = "gpt-3.5-turbo-0613",
      retry: int = 5,
      wait: int = 5,
      key: str = None,
      verbose: bool = True,
      system_prompt: str = None,
      temperature: float = 0,
      timeout: int = 60,
      api_base: str = "OFFICIAL",
      max_tokens: int = 1024,
      img_size: int = 512,
      img_detail: str = "low",
      **kwargs,
  ):
    self.model = model
    self.cur_idx = 0
    self.fail_msg = "Failed to obtain answer via API. "
    self.max_tokens = max_tokens
    self.temperature = temperature

    openai_key = os.environ.get("OPENAI_API_KEY", None) if key is None else key
    self.openai_key = openai_key
    assert img_size > 0 or img_size == -1
    self.img_size = img_size
    assert img_detail in ["high", "low"]
    self.img_detail = img_detail

    self.vision = False
    if model == "gpt-4-vision-preview":
      self.vision = True
    self.timeout = timeout

    assert (
        isinstance(openai_key, str) and openai_key.startswith("sk-")
    ), f"Illegal openai_key {openai_key}. Please set the environment variable OPENAI_API_KEY to your openai key. "
    super().__init__(
        wait=wait,
        retry=retry,
        system_prompt=system_prompt,
        verbose=verbose,
        **kwargs,
    )

    if api_base in APIBASES:
      self.api_base = APIBASES[api_base]
    elif api_base.startswith("http"):
      self.api_base = api_base
    else:
      self.logger.error("Unknown API Base. ")
      sys.exit(-1)

  # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
  # content can be a string or a list of image & text
  def prepare_inputs(self, inputs):
    input_msgs = []
    if self.system_prompt is not None:
      input_msgs.append(dict(role="system", content=self.system_prompt))
    if isinstance(inputs, str):
      input_msgs.append(dict(role="user", content=inputs))
      return input_msgs
    assert isinstance(inputs, list)
    dict_flag = [isinstance(x, dict) for x in inputs]
    if np.all(dict_flag):
      input_msgs.extend(inputs)
      return input_msgs
    str_flag = [isinstance(x, str) for x in inputs]
    if np.all(str_flag):
      img_flag = [x.startswith("http") or osp.exists(x) for x in inputs]
      if np.any(img_flag):
        content_list = []
        for fl, msg in zip(img_flag, inputs):
          if not fl:
            content_list.append(dict(type="text", text=msg))
          elif msg.startswith("http"):
            content_list.append(
                dict(
                    type="image_url",
                    image_url={
                        "url": msg,
                        "detail": self.img_detail
                    },
                ))
          elif osp.exists(msg):
            img = Image.open(msg)
            b64 = encode_image_to_base64(img, target_size=self.img_size)
            img_struct = dict(url=f"data:image/jpeg;base64,{b64}",
                              detail=self.img_detail)
            content_list.append(dict(type="image_url", image_url=img_struct))
        input_msgs.append(dict(role="user", content=content_list))
        return input_msgs
      else:
        raise ValueError("The prompt does not contain an image.")
    raise NotImplementedError("list of list prompt not implemented now. ")

  def generate_inner(self, inputs, **kwargs) -> str:
    input_msgs = self.prepare_inputs(inputs)
    temperature = kwargs.pop("temperature", self.temperature)
    max_tokens = kwargs.pop("max_tokens", self.max_tokens)

    context_window = GPT_context_window(self.model)
    max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
    if 0 < max_tokens <= 100:
      self.logger.warning(
          "Less than 100 tokens left, may exceed the context window with some additional meta symbols. "
      )
    if max_tokens <= 0:
      return (
          0,
          self.fail_msg + "Input string longer than context window. ",
          "Length Exceeded. ",
      )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.openai_key}",
    }
    payload = dict(
        model=self.model,
        messages=input_msgs,
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
        **kwargs,
    )
    print("model", self.model)
    print("messages", input_msgs)
    response = requests.post(
        self.api_base,
        headers=headers,
        data=json.dumps(payload),
        timeout=self.timeout * 1.1,
    )
    ret_code = response.status_code
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
    answer = self.fail_msg
    try:
      resp_struct = json.loads(response.text)
      answer = resp_struct["choices"][0]["message"]["content"].strip()
      print("answer", answer)
    except:  # noqa: E722
      pass
    return ret_code, answer, response

  def get_token_len(self, inputs) -> int:
    import tiktoken

    enc = tiktoken.encoding_for_model(self.model)
    if isinstance(inputs, str):
      if inputs.startswith("http") or osp.exists(inputs):
        return 65 if self.img_detail == "low" else 130
      else:
        return len(enc.encode(inputs))
    elif isinstance(inputs, dict):
      assert "content" in inputs
      return self.get_token_len(inputs["content"])
    assert isinstance(inputs, list)
    res = 0
    for item in inputs:
      res += self.get_token_len(item)
    return res


class GPT4V(OpenAIWrapper):

  def __init__(self, *args, **kwargs):
    self.prompter = kwargs.pop("prompter")
    self.task_prompt = self.prompter.task_prompt

    super(GPT4V, self).__init__(*args, **kwargs)

  def generate(self, image_path):
    assert self.model == "gpt-4-vision-preview"
    return super(GPT4V, self).generate([self.task_prompt, image_path])

  def multi_generate(self, image_paths, prompt, dataset=None):
    assert self.model == "gpt-4-vision-preview"
    return super(GPT4V, self).generate(image_paths + [prompt])

  def interleave_generate(self, ti_list, dataset=None):
    assert self.model == "gpt-4-vision-preview"
    return super(GPT4V, self).generate(ti_list)


class GPT4V1Shot(GPT4V):

  def __init__(self, *args, **kwargs):
    self.prompter = kwargs.pop("prompter")
    self.task_prompt = self.prompter.task_prompt
    self.one_shot_image_path = self.prompter.one_shot_image_path
    self.one_shot_answer = self.prompter.one_shot_answer

    super(GPT4V, self).__init__(*args, **kwargs)

  def generate(self, image_path):
    assert self.model == "gpt-4-vision-preview"

    prompts = []
    prompts.append(self.task_prompt + "\nFor example, for the following image:")
    prompts.append(self.one_shot_image_path)
    prompts.append(self.one_shot_answer)
    prompts.append("\nNow do the following one:")
    prompts.append(image_path)
    return super(GPT4V, self).generate(prompts)


class GeminiWrapper(BaseAPI):
  is_api: bool = True

  def __init__(
      self,
      retry: int = 5,
      wait: int = 5,
      key: str = None,
      verbose: bool = True,
      temperature: float = 0.0,
      system_prompt: str = None,
      max_tokens: int = 1024,
      **kwargs,
  ):
    self.fail_msg = "Failed to obtain answer via API. "
    self.max_tokens = max_tokens
    self.temperature = temperature
    if key is None:
      key = os.environ.get("GOOGLE_API_KEY", None)
    assert key is not None
    genai.configure(api_key=key)
    super().__init__(
        wait=wait,
        retry=retry,
        system_prompt=system_prompt,
        verbose=verbose,
        **kwargs,
    )

  @staticmethod
  def build_msgs(msgs_raw, system_prompt=None):
    msgs = cp.deepcopy(msgs_raw)
    assert len(msgs) % 2 == 1

    if system_prompt is not None:
      msgs[0] = [system_prompt, msgs[0]]
    ret = []
    for i, msg in enumerate(msgs):
      role = "user" if i % 2 == 0 else "model"
      parts = msg if isinstance(msg, list) else [msg]
      ret.append(dict(role=role, parts=parts))
    return ret

  def generate_inner(self, inputs, **kwargs) -> str:
    assert isinstance(inputs, str) or isinstance(inputs, list)
    pure_text = True
    if isinstance(inputs, list):
      for pth in inputs:
        if osp.exists(pth) or pth.startswith("http"):
          pure_text = False
    model = (genai.GenerativeModel("gemini-pro")
             if pure_text else genai.GenerativeModel("gemini-pro-vision"))
    if isinstance(inputs, str):
      messages = ([inputs] if self.system_prompt is None else
                  [self.system_prompt, inputs])
    elif pure_text:
      messages = self.build_msgs(inputs, self.system_prompt)
    else:
      messages = [] if self.system_prompt is None else [self.system_prompt]
      for s in inputs:
        if osp.exists(s):
          messages.append(Image.open(s))
        elif s.startswith("http"):
          pth = download_file(s)
          messages.append(Image.open(pth))
          shutil.remove(pth)
        else:
          messages.append(s)
    gen_config = dict(max_output_tokens=self.max_tokens,
                      temperature=self.temperature)
    gen_config.update(self.kwargs)
    try:
      answer = model.generate_content(
          messages,
          generation_config=genai.types.GenerationConfig(**gen_config)).text
      print("message", messages)
      print("answer", answer)
      return 0, answer, "Succeeded! "
    except Exception as err:
      if self.verbose:
        self.logger.error(err)
        self.logger.error(f"The input messages are {inputs}.")

      return -1, "", ""


class GeminiProVision(GeminiWrapper):

  def __init__(self, *args, **kwargs):
    self.prompter = kwargs.pop("prompter")
    self.task_prompt = self.prompter.task_prompt

    super(GeminiProVision, self).__init__(*args, **kwargs)

  def generate(self, image_path, prompt, dataset=None):
    del prompt
    # return super(GeminiProVision, self).generate([image_path, prompt])
    return super(GeminiProVision, self).generate([self.task_prompt, image_path])

  def multi_generate(self, image_paths, prompt, dataset=None):
    return super(GeminiProVision, self).generate(image_paths + [prompt])

  def interleave_generate(self, ti_list, dataset=None):
    return super(GeminiProVision, self).generate(ti_list)


class GeminiProVision1Shot(GeminiProVision):

  def __init__(self, *args, **kwargs):
    self.prompter = kwargs.pop("prompter")
    self.task_prompt = self.prompter.task_prompt
    self.one_shot_image_path = self.prompter.one_shot_image_path
    self.one_shot_answer = self.prompter.one_shot_answer

    super(GeminiProVision, self).__init__(*args, **kwargs)

  def generate(self, image_path, prompt, dataset=None):
    del prompt

    prompts = []
    prompts.append(self.task_prompt + "\nFor example, for the following image:")
    prompts.append(self.one_shot_image_path)
    prompts.append(self.one_shot_answer)
    prompts.append("\nNow do the following one:")
    prompts.append(image_path)
    return super(GeminiProVision, self).generate(prompts)


all_models = {
    "GPT4V":
        partial(
            GPT4V,
            model="gpt-4-vision-preview",
            temperature=0,
            img_size=512,
            img_detail="low",
            retry=10,
        ),
    "GPT4V1Shot":
        partial(
            GPT4V1Shot,
            model="gpt-4-vision-preview",
            temperature=0,
            img_size=512,
            img_detail="low",
            retry=10,
        ),
    'GeminiProVision':
        partial(GeminiProVision, temperature=0.1, retry=10),
    'GeminiProVision1Shot':
        partial(GeminiProVision1Shot, temperature=0.1, retry=10),
}
