import os.path as osp
import base64, os, logging, io
from uuid import uuid4
from tqdm import tqdm
from PIL import Image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
  image = decode_base64_to_image(base64_string, target_size=target_size)
  image.save(image_path)


def encode_image_to_base64(img, target_size=-1):
  # if target_size == -1, will not do resizing
  # else, will set the max_size ot (target_size, target_size)
  if img.mode in ("RGBA", "P"):
    img = img.convert("RGB")
  tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
  if target_size > 0:
    img.thumbnail((target_size, target_size))
  img.save(tmp)
  with open(tmp, 'rb') as image_file:
    image_data = image_file.read()
  ret = base64.b64encode(image_data).decode('utf-8')
  os.remove(tmp)
  return ret


def decode_base64_to_image(base64_string, target_size=-1):
  image_data = base64.b64decode(base64_string)
  image = Image.open(io.BytesIO(image_data))
  if image.mode in ('RGBA', 'P'):
    image = image.convert('RGB')
  if target_size > 0:
    image.thumbnail((target_size, target_size))
  return image


logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
  logger = logging.getLogger(name)
  if name in logger_initialized:
    return logger

  for logger_name in logger_initialized:
    if name.startswith(logger_name):
      return logger

  stream_handler = logging.StreamHandler()
  handlers = [stream_handler]

  try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
      rank = dist.get_rank()
    else:
      rank = 0
  except ImportError:
    rank = 0

  if rank == 0 and log_file is not None:
    file_handler = logging.FileHandler(log_file, file_mode)
    handlers.append(file_handler)

  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  for handler in handlers:
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    logger.addHandler(handler)

  if rank == 0:
    logger.setLevel(log_level)
  else:
    logger.setLevel(logging.ERROR)

  logger_initialized[name] = True
  return logger


def read_ok(img_path):
  if not osp.exists(img_path):
    return False
  try:
    im = Image.open(img_path)
    assert im.size[0] > 0 and im.size[1] > 0
    return True
  except:
    return False


def download_file(url, filename=None):
  import urllib.request

  class DownloadProgressBar(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
      if tsize is not None:
        self.total = tsize
      self.update(b * bsize - self.n)

  if filename is None:
    filename = url.split('/')[-1]

  with DownloadProgressBar(unit='B',
                           unit_scale=True,
                           miniters=1,
                           desc=url.split('/')[-1]) as t:
    urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
  return filename
