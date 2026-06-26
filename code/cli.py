from pix2tex.dataset.transforms import test_transform
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import atexit
from contextlib import suppress
import logging
import yaml
import re
from datetime import datetime

with suppress(ImportError, AttributeError):
    import readline

import numpy as np
import torch
from torch._appdirs import user_data_dir
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from pix2tex.dataset.latex2png import tex2pil
from pix2tex.models import get_model
from pix2tex.utils import *
from pix2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints
from pix2tex.xai.viz import save_attention_overlays
from pix2tex.xai.gradcam import add_gradcam_to_trace
from pix2tex.xai.trace import attention_diffuseness, confidence_summary
from pix2tex.xai.consistency import attribution_consistency_score

import contextlib as _contextlib
import pix2tex.utils as _pix2tex_utils

@_contextlib.contextmanager
def in_model_path():
    """Like pix2tex.utils.in_model_path but respects _OVERRIDE_MODEL_DIR."""
    import cli as _self_module
    target = _self_module._OVERRIDE_MODEL_DIR
    if target is None:
        with _pix2tex_utils.in_model_path():
            yield
    else:
        saved = os.getcwd()
        os.chdir(target)
        try:
            yield
        finally:
            os.chdir(saved)


# Set this before instantiating LatexOCR to load weights from a custom directory
# instead of the pix2tex package install (e.g. "models/pix2tex_baseline").
_OVERRIDE_MODEL_DIR: Optional[str] = None


def minmax_size(img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:

    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


class LatexOCR:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)
        self.model.args._xai_tokenizer = self.tokenizer

    def _decode_with_trace(self, im: torch.Tensor, temperature: float):
        return self.model.generate_with_trace(im.to(self.args.device), temperature=temperature)

    def _quality_from_trace(self, trace: dict, im: torch.Tensor):
        conf = trace.get('confidences', None)
        attn = trace.get('cross_attentions', None)
        conf_stats = confidence_summary(conf)
        diff = 1.0
        if torch.is_tensor(attn) and attn.numel() > 0:
            diffs = []
            steps = attn.shape[1] if attn.ndim >= 3 else 0
            for t in range(steps):
                diffs.append(attention_diffuseness(attn[0, t]).item())
            diff = float(sum(diffs) / max(len(diffs), 1))

        dec = trace.get('tokens', None)
        token_ids = None
        if torch.is_tensor(dec):
            token_ids = dec[0].detach().cpu().tolist() if dec.ndim == 2 else dec.detach().cpu().tolist()

        consistency = 0.0
        if token_ids is not None and self.tokenizer is not None:
            consistency = attribution_consistency_score(
                image_tensor=im[0].detach().cpu(),
                token_ids=token_ids,
                token_maps=attn,
                tokenizer=self.tokenizer,
                patch_size=self.args.get('patch_size', 16),
            )

        score = conf_stats['mean'] - 0.5 * diff + 0.25 * consistency
        return {
            'score': score,
            'confidence_mean': conf_stats['mean'],
            'confidence_min': conf_stats['min'],
            'diffuseness': diff,
            'consistency': consistency,
        }

    def _retry_temperatures(self, base_temperature: float):
        values = self.args.get('quality_retry_temperatures', None)
        if isinstance(values, str):
            vals = []
            for part in values.split(','):
                part = part.strip()
                if not part:
                    continue
                with suppress(ValueError):
                    vals.append(float(part))
            values = vals
        if isinstance(values, (int, float)):
            values = [float(values)]
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            values = [max(0.05, base_temperature * 0.5), max(0.05, base_temperature * 0.75)]
        out = []
        for t in values:
            with suppress(TypeError, ValueError):
                v = float(t)
                if v > 0:
                    out.append(v)
        return out

    def _should_trigger_redecode(self, quality: dict):
        conf_th = float(self.args.get('quality_confidence_threshold', 0.45))
        diff_th = float(self.args.get('quality_diffuseness_threshold', 0.78))
        cons_th = float(self.args.get('quality_consistency_threshold', 0.02))

        low_conf = quality.get('confidence_mean', 0.0) < conf_th
        high_diff = quality.get('diffuseness', 1.0) > diff_th
        low_cons = quality.get('consistency', 0.0) < cons_th
        return bool(low_conf and (high_diff or low_cons))

    @in_model_path()
    def __call__(self, img=None, resize=True):
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        trace = None
        quality = None
        explain_enabled = bool(self.args.get('explain', False))
        gate_enabled = bool(self.args.get('quality_gate', False))
        temperature = float(self.args.get('temperature', .25))

        if explain_enabled or gate_enabled:
            trace = self._decode_with_trace(im, temperature)
            with suppress(Exception):
                quality = self._quality_from_trace(trace, im)
                trace['quality'] = quality

            retries = []
            used_retry = False
            if gate_enabled and quality is not None and self._should_trigger_redecode(quality):
                retry_temps = self._retry_temperatures(temperature)
                max_retries = int(self.args.get('quality_max_retries', 1))
                retry_temps = retry_temps[:max(max_retries, 0)]

                candidates = [(temperature, trace, quality)]
                for t_retry in retry_temps:
                    retry_trace = self._decode_with_trace(im, t_retry)
                    retry_quality = self._quality_from_trace(retry_trace, im)
                    retry_trace['quality'] = retry_quality
                    retries.append({'temperature': float(t_retry), **retry_quality})
                    candidates.append((t_retry, retry_trace, retry_quality))

                best_temp, best_trace, best_quality = max(candidates, key=lambda x: x[2].get('score', -1e9))
                if best_trace is not trace:
                    trace = best_trace
                    quality = best_quality
                    used_retry = True

                if quality is not None:
                    quality['redecode_triggered'] = True
                    quality['redecode_used'] = bool(used_retry)
                    quality['selected_temperature'] = float(best_temp)
                    quality['retry_attempts'] = retries
                    trace['quality'] = quality

            if quality is not None and 'redecode_triggered' not in quality:
                quality['redecode_triggered'] = False
                quality['redecode_used'] = False
                quality['selected_temperature'] = float(temperature)
                quality['retry_attempts'] = []
                trace['quality'] = quality

            if explain_enabled and self.args.get('gradcam', False):
                with suppress(Exception):
                    trace = add_gradcam_to_trace(self.model, im.to(self.args.device), trace, max_tokens=self.args.get('xai_max_tokens', 8))
            dec = trace['tokens']
        else:
            dec = self.model.generate(im.to(self.args.device), temperature=temperature)

        pred = post_process(token2str(dec, self.tokenizer)[0])

        if trace is not None and self.args.get('save_xai_dir', None):
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            out_dir = os.path.join(self.args.save_xai_dir, f'xai_{stamp}')
            try:
                save_attention_overlays(
                    image_tensor=im[0].detach().cpu(),
                    trace=trace,
                    tokenizer=self.tokenizer,
                    output_dir=out_dir,
                    patch_size=self.args.get('patch_size', 16),
                    max_tokens=self.args.get('xai_max_tokens', 8),
                )
                print(f'[XAI] saved artifacts to {out_dir}')
            except Exception as e:
                print(f'[XAI] error saving artifacts: {type(e).__name__}: {e}')

        try:
            clipboard.copy(pred)
        except:
            pass
        if trace is not None:
            return pred, trace
        return pred


def output_prediction(pred, args):
    TERM = os.getenv('TERM', 'xterm')
    if not sys.stdout.isatty():
        TERM = 'dumb'
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import get_formatter_by_name

        if TERM.split('-')[-1] == '256color':
            formatter_name = 'terminal256'
        elif TERM != 'dumb':
            formatter_name = 'terminal'
        else:
            formatter_name = None
        if formatter_name:
            formatter = get_formatter_by_name(formatter_name)
            lexer = get_lexer_by_name('tex')
            print(highlight(pred, lexer, formatter), end='')
    except ImportError:
        TERM = 'dumb'
    if TERM == 'dumb':
        print(pred)
    if args.show or args.katex:
        try:
            if args.katex:
                raise ValueError
            tex2pil([f'$${pred}$$'])[0].show()
        except Exception as e:
            # render using katex
            import webbrowser
            from urllib.parse import quote
            url = 'https://katex.org/?data=' + \
                quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
"strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
            webbrowser.open(url)


def predict(model, file, arguments):
    img = None
    if file:
        try:
            img = Image.open(os.path.expanduser(file))
        except Exception as e:
            print(e, end='')
    else:
        try:
            img = ImageGrab.grabclipboard()
        except NotImplementedError as e:
            print(e, end='')
    result = model(img)
    if isinstance(result, tuple):
        pred = result[0]
        trace = result[1]
        quality = trace.get('quality', {})
        if quality:
            print('[XAI] conf=%.3f diffuse=%.3f consistency=%.3f score=%.3f redecode=%s temp=%.3f' % (
                quality.get('confidence_mean', 0.0),
                quality.get('diffuseness', 1.0),
                quality.get('consistency', 0.0),
                quality.get('score', -1.0),
                'yes' if quality.get('redecode_used', False) else 'no',
                quality.get('selected_temperature', arguments.temperature),
            ))
    else:
        pred = result
    output_prediction(pred, arguments)

def check_file_path(paths:List[Path], wdir:Optional[Path]=None)->List[str]:
    files = []
    for path in paths:
        if type(path)==str:
            if path=='':
                continue
            path=Path(path)
        pathsi = ([path] if wdir is None else [path, wdir/path])
        for p in pathsi:
            if p.exists():
                files.append(str(p.resolve()))
            elif '*' in path.name:
                files.extend([str(pi.resolve()) for pi in p.parent.glob(p.name)])
    return list(set(files))

def main(arguments):
    path = user_data_dir('pix2tex')
    os.makedirs(path, exist_ok=True)
    history_file = os.path.join(path, 'history.txt')
    with suppress(NameError):
        # user can `ln -s /dev/null ~/.local/share/pix2tex/history.txt` to
        # disable history record
        with suppress(OSError):
            readline.read_history_file(history_file)
        atexit.register(readline.write_history_file, history_file)
    files = check_file_path(arguments.file)
    wdir = Path(os.getcwd())
    
    # Convert relative save_xai_dir to absolute before model initialization
    if arguments.save_xai_dir is not None and not os.path.isabs(arguments.save_xai_dir):
        arguments.save_xai_dir = os.path.abspath(arguments.save_xai_dir)
    with in_model_path():
        model = LatexOCR(arguments)
        if files:
            for file in check_file_path(arguments.file, wdir):
                print(file + ': ', end='')
                predict(model, file, arguments)
                model.last_pic = None
                with suppress(NameError):
                    readline.add_history(file)
            exit()
        pat = re.compile(r't=([\.\d]+)')
        while True:
            try:
                instructions = input('Predict LaTeX code for image ("h" for help). ')
            except KeyboardInterrupt:
                # TODO: make the last line gray
                print("")
                continue
            except EOFError:
                break
            file = instructions.strip()
            ins = file.lower()
            t = pat.match(ins)
            if ins == 'x':
                break
            elif ins in ['?', 'h', 'help']:
                print('''pix2tex help:

    Usage:
        On Windows and macOS you can copy the image into memory and just press ENTER to get a prediction.
        Alternatively you can paste the image file path here and submit.

        You might get a different prediction every time you submit the same image. If the result you got was close you
        can just predict the same image by pressing ENTER again. If that still does not work you can change the temperature
        or you have to take another picture with another resolution (e.g. zoom out and take a screenshot with lower resolution). 

        Press "x" to close the program.
        You can interrupt the model if it takes too long by pressing Ctrl+C.

    Visualization:
        You can either render the code into a png using XeLaTeX (see README) to get an image file back.
        This is slow and requires a working installation of XeLaTeX. To activate type 'show' or set the flag --show
        Alternatively you can render the expression in the browser using katex.org. Type 'katex' or set --katex

    Settings:
        to toggle one of these settings: 'show', 'katex', 'no_resize' just type it into the console
        Change the temperature (default=0.333) type: "t=0.XX" to set a new temperature.
                    ''')
                continue
            elif ins in ['show', 'katex', 'no_resize']:
                setattr(arguments, ins, not getattr(arguments, ins, False))
                print('set %s to %s' % (ins, getattr(arguments, ins)))
                continue
            elif t is not None:
                t = t.groups()[0]
                model.args.temperature = float(t)+1e-8
                print('new temperature: T=%.3f' % model.args.temperature)
                continue
            files = check_file_path(file.split(' '), wdir)
            with suppress(KeyboardInterrupt):
                if files:
                    for file in files:
                        if len(files)>1:
                            print(file + ': ', end='')
                        predict(model, file, arguments)
                else:
                    predict(model, file, arguments)
            file = None
