import torch
from huggingface_hub import upload_file
import onnx
import onnxruntime as ort
import torch.nn as nn
import time
import gc
from pathlib import Path
import tqdm

import data_utils
from models.swinunet.vision_transformer import SwinUnet
from models.nnwnet.nnwnet import WNet2D
from models.msu_unet.msu_unet import MSU_Net
from models.cenet.net import CENet
from models.unet_plus_plus.unet_plus_plus import NestedUNet
from models.cstanet.CSTANet import CSTANet
from models.ce_net.ce_net import CE_Net
from models.uresnet.uresnet import UResNet
from models.attention_unet.attention_unet import AttentionUNet
from models.r2unet.r2unet import R2UNet
from models.wnet.wnet import WNet
from models.iternet.iternet import Iternet
from models.rrwnet.rrwnet_model import RRWNet
from models.rvbsin.rvbsin import VesselSegNetwork
from models.custom_unet.unet import UNet

def export_jit(model, path, input_shape=(1, 1, 512, 512)):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(path)

def export_onnx(model, path, input_shape=(1, 1, 512, 512)):
    model.eval()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model.cpu(), dummy_input, path, 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

def load_onnx_model(path, device='cuda'):
    # Load the ONNX model
    model = onnx.load(path)
    
    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Create an ONNX Runtime session
    session = ort.InferenceSession(path, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])

    return session

def _get_first_layer(m):
    "Access first layer of a model"
    c,p,n = m,None,None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p,c=c,getattr(c,n)
    return c,p,n

def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    print(f"Previous layer weights shape: {previous_layer.weight.data.shape}, new layer weights shape: {new_layer.weight.data.shape}")
    if n_in==1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
        print(f"Previous layer weights shape: {previous_layer.weight.data.shape}, new layer weights shape: {new_layer.weight.data.shape}")
    elif n_in==2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:2,:] * 1.5
    else:
        # keep 3 channels weights and set others to null
        print(f"Warning: More than 3 input channels, only first 3 channels will be initialized with pretrained weights, others will be set to zero.")
        new_layer.weight.data[:2,:] = previous_layer.weight[:2,:].data
        new_layer.weight.data[2:,:].zero_()

def _update_conv_layer(layer, param_str, param, pretrained):
    "Change layer based on parameter"
    assert isinstance(layer, nn.Conv2d), f'Change only supported with Conv2d, found {layer.__class__.__name__}'
    params = {attr:getattr(layer, attr) for attr in 'in_channels out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(layer, 'bias') is not None
    params[param_str] = param
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, layer)
    return new_layer

def _update_first_layer_input(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    new_layer = _update_conv_layer(first_layer, 'in_channels', n_in, pretrained)
    setattr(parent, name, new_layer)

def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)

def upload_file_to_hf(repo_id, hf_model_name, file_path):
    repo_id = "DigitalHolography"
    huggingface_model_name = "nnwnet_av_corr_diasys"
    repo_id = f"{repo_id}/{huggingface_model_name}"
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=huggingface_model_name,
        repo_id=repo_id,
        repo_type="model"
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_tensor, device='cuda', iterations=100):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    # Warm-up (important for GPU)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()  # ensure all ops are finished
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations
    print(f"Average inference time per run: {avg_time * 1000:.3f} ms")
    return avg_time

def measure_onnx_inference_time(session, input_tensor, iterations=100, warmup=10):
    # Convert input tensor to numpy
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()

    # Get input name for ONNX session
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_tensor})

    # Timing
    start = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_tensor})
    end = time.time()

    avg_time = (end - start) / iterations
    print(f"Average ONNX inference time per run: {avg_time * 1000:.3f} ms")
    return avg_time

def count_onnx_parameters(onnx_path):
    model = onnx.load(onnx_path)
    param_count = 0

    for tensor in model.graph.initializer:
        param_array = onnx.numpy_helper.to_array(tensor)
        param_count += param_array.size

    print(f"Total number of parameters: {param_count:,}")
    return param_count

class ModelWrapper:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, xb):
        with torch.no_grad():
            return self.model(xb)

    def inference_time(self, input_tensor):
        return measure_inference_time(self.model, input_tensor, device=self.device)

    def num_parameters(self):
        return count_parameters(self.model)
    
    def forward(self, xb):
        return self.predict(xb)

class ONNXModel(ModelWrapper):
    def __init__(self, path, device='cuda'):
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.path = path

    def predict(self, xb):
        xb_np = xb.detach().cpu().numpy()
        pred = self.session.run([self.output_name], {self.input_name: xb_np})[0]
        return torch.tensor(pred, device=xb.device)

    def inference_time(self, input_tensor):
        return measure_onnx_inference_time(self.session, input_tensor)

    def num_parameters(self):
        return count_onnx_parameters(self.path)
    
class TorchScriptModel(ModelWrapper):
    def __init__(self, path, device='cuda'):
        self.model = torch.jit.load(path).to(device).eval()
        self.path = path

    def predict(self, xb):
        with torch.no_grad():
            return self.model(xb)

    def inference_time(self, input_tensor):
        return measure_inference_time(self.model, input_tensor)

    def num_parameters(self):
        return count_parameters(self.model)



class StateDictModel:
    def __init__(self, path, model_fn, in_channels=1, n_classes=1, device='cuda'):
        self.model = model_fn.init_from_state_dict(in_channels=in_channels, n_classes=n_classes, weight_file=path).to(device).eval()

    def predict(self, xb):
        with torch.no_grad():
            return self.model(xb)

    def inference_time(self, input_tensor):
        return measure_inference_time(self.model, input_tensor)

    def num_parameters(self):
        return count_parameters(self.model)
    

def get_model_class(model_name):
    model_name = model_name.lower()
    model_name_to_class = {
        'unet++': NestedUNet,
        'wnet2d': WNet2D,
        'swinunet': SwinUnet,
        'msunet': MSU_Net,
        'cenet': CENet,
        'r2unet': R2UNet,
        'rvb_sin': VesselSegNetwork,
        'ce_net': CE_Net,
        'rrwnet': RRWNet,
        'iternet': Iternet,
        'attentionunet': AttentionUNet,
        'cstanet': CSTANet,
        'wnet': WNet,
        'uresnet': UResNet,
        'unet': UNet,
    }
    for key, cls in model_name_to_class.items():
        if key in model_name:
            return cls
    raise ValueError(f"Unknown model type for name: {model_name}")

def predict_and_show(model, val_loader, cmap='viridis', multi=None, n=20):
    # predict masks
    masks_pred = []
    inputs = []
    targets = []
    multi = multi
    for input, target in iter(val_loader):
        mask = model.predict(input.cuda())
        multi = mask.shape[1] > 1

        mask = torch.sigmoid(mask)
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1

        inputs.append(input.squeeze(0).cpu().numpy())
        masks_pred.append(mask.squeeze(0).cpu().detach().numpy())
        targets.append(target.squeeze(0).cpu().numpy())
    data_utils.show_masks(inputs, targets, masks_pred, multi=multi, cmap=cmap, n=n)

def evaluate_model(
    model,
    model_name,
    dataloader,
    metrics,
    device='cuda',
    per_class_metrics=False,
    per_sample_metrics=False,
    n_classes=2,
    show_results=False,
    n=3
):
    """ Evaluate a model on a dataloader using specified metrics.
    Args:
        model: The model to evaluate.
        model_name (str): Name of the model.
        dataloader (DataLoader): Dataloader for evaluation.
        metrics (List[Tuple[str, callable]]): List of (name, metric_fn) pairs.
        device (str): 'cpu' or 'cuda'.
        per_class_metrics (bool): Whether to compute metrics per class. Used for class imbalance analysis.
        per_sample_metrics (bool): Whether to compute metrics per sample. Used for outlier detection.
        n_classes (int): Number of classes.
        show_results (bool): Whether to display evaluation results.
        n (int): Number of samples to display if show_results is True.
    Returns:
        Dict: Average metrics and optionally per-sample results.
    """
    metric_sums = {}
    if per_class_metrics:
        for name, _ in metrics:
            for c in range(n_classes):
                metric_sums[f"{name}_class_{c}"] = 0.0
    else:
        metric_sums = {name: 0.0 for name, _ in metrics}

    num_samples = 0
    per_sample_results = []

    for xb, yb in tqdm.tqdm(dataloader):
        xb, yb = xb.to(device), yb.to(device)

        pred = model.predict(xb)

        yb = torch.Tensor(yb)
        pred = torch.Tensor(pred)

        for i in range(xb.shape[0]):
            num_samples += 1

            if per_sample_metrics:
                sample_entry = {
                    "model": model_name,
                    "sample_idx": num_samples - 1
                }

            for name, fn in metrics:
                if per_class_metrics:
                    res = fn(pred[i:i+1], yb[i:i+1], return_per_class=True)

                    for c in range(n_classes):
                        key = f"{name}_class_{c}"
                        metric_sums[key] += res[c]

                        if per_sample_metrics:
                            sample_entry[key] = res[c]
                else:
                    val = fn(pred[i:i+1], yb[i:i+1]).item()
                    metric_sums[name] += val

                    if per_sample_metrics:
                        sample_entry[name] = val

            if per_sample_metrics:
                per_sample_results.append(sample_entry)

    avg_metrics = {k: v / num_samples for k, v in metric_sums.items()}
    avg_metrics["model"] = model_name

    # timing + params
    input_tensor = torch.randn_like(xb[:1])
    avg_metrics["inference_time"] = model.inference_time(input_tensor)
    avg_metrics["num_parameters"] = model.num_parameters()

    if show_results:
        x, y = next(iter(dataloader))
        multi = x.shape[1] > 1
        print(f"Model: {model_name}")
        print(f"{multi=}")
        predict_and_show(model, dataloader, n=n, cmap='gray', multi=multi)

    if per_sample_metrics:
        return avg_metrics, per_sample_results

    return avg_metrics


def evaluate_models(
    model_paths,
    dataloader,
    metrics,
    input_channels=1,
    num_classes=2,
    device='cuda',
    results=[],
    per_class_metrics=False,
    per_sample_metrics=False,
    show_results=False,
    n=3,
    extension = ".onnx",
):
    """ Evaluate multiple models and return their metrics.
    Args:
        model_paths (List[str]): List of paths to model files.
        dataloader (DataLoader): Dataloader for evaluation.
        metrics (List[Tuple[str, callable]]): List of (name, metric_fn) pairs.
        input_channels (int): Number of input channels for the models.
        num_classes (int): Number of output classes for the models.
        device (str): 'cpu' or 'cuda'.
        results (List[Dict]): List to append results to. If empty, a new list will be created.
        per_class_metrics (bool): Whether to compute metrics per class. Used for class imbalance analysis.
        per_sample_metrics (bool): Whether to compute metrics per sample. Used for outlier detection.
        show_results (bool): Whether to display evaluation results.
        n (int): Number of samples to display if show_results is True.
        extension (str): File extension of the model files (e.g., ".onnx", ".pt").
    Returns:
        List[Dict]: List of average metrics for each model and optionally per-sample results.
    """
    all_per_sample = []

    for model_path in model_paths:
        model = load_model(model_path, extension=extension, in_channels=input_channels, num_classes=num_classes, device=device)[1]
        model_name = Path(model_path).stem

        print(f"Evaluating {model_name}")

        if per_sample_metrics:
            model_metrics, per_sample = evaluate_model(model, model_name, dataloader, metrics, device=device, per_class_metrics=per_class_metrics, per_sample_metrics=per_sample_metrics, n_classes=num_classes, show_results=show_results, n=n)
            all_per_sample.extend(per_sample)
        else:
            model_metrics = evaluate_model(model, model_name, dataloader, metrics, device=device, per_class_metrics=per_class_metrics, per_sample_metrics=False, n_classes=num_classes, show_results=show_results, n=n)
        results.append(model_metrics)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if per_sample_metrics:
        return results, all_per_sample
    return results

def load_model(model_path, extension, in_channels, num_classes, device='cuda'):
    model_name = Path(model_path).stem
    if extension == ".onnx":
        return model_name, ONNXModel(model_path, device)

    elif extension == ".pt":
        return model_name, TorchScriptModel(model_path, device)

    elif extension == ".pth":
        model_fn = get_model_class(model_name)
        if model_fn is None:
            raise ValueError("model_fn required for state_dict")
        return model_name, StateDictModel(model_path, model_fn, in_channels, num_classes, device)
    else:
        raise ValueError(f"Unsupported model extension: {extension}")

def load_models(model_paths, extension, in_channels, num_classes, device='cuda'):
    models = []

    for path in model_paths:
        models.append(load_model(path, extension, in_channels, num_classes, device))

    return models

def get_filtered_model_paths(folder, extension=".onnx", keys=None):
    all_paths = list(Path(folder).glob(f"*{extension}"))
    if keys:
        filtered = []
        for p in all_paths:
            filename = p.name  # Just the filename, e.g., 'uresnet_model.onnx'
            for k in keys:
                if k in filename:
                    print(f"Model {filename} matches key: {k}")
                    filtered.append(str(p))
        return filtered
    return [str(p) for p in all_paths]
