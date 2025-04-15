import logging
from os.path import isabs, join, expanduser, exists
from os import getcwd, makedirs
import pandas as pd
import hydra
import zeus.monitor.energy
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import DataLoader
from zeus.monitor import ZeusMonitor, Measurement

from qtransform.model import QTRModelWrapper
import os
import torch
from torch.utils import data as torch_data  # prevent naming conflict with data from dataloaders
from datetime import datetime
import time

from qtransform.run import generate
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform import device_singleton
from qtransform.model import get_model_wrapper, DynamicCheckpointQTRModelWrapper
from functools import lru_cache

log = logging.getLogger(__name__)


# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


def run(cfg: DictConfig):
    """ launches energy benchmark with provided config"""
    log.info("================")
    log.info("Running energy benchmark")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")

    if "dataloader" not in cfg.dataset:
        log.error(
            f"dataloder not specified for dataset: {cfg.dataset.name}. Use dataset=huggingface to get one automaticly.")
    device_singleton.device = cfg.device
    device = device_singleton.device
    torch.manual_seed(cfg.seed)

    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")
    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:
        if not model_wrapper.quantized:
            log.info(f'Quantizing model')
            model_wrapper.quantize_model(quant_cfg)
        else:
            warn_once(log, f'Model was already quantized, ignoring quant_cfg from hydra')
        # from qtransform.quantization import get_quantizer
        # quantizer, model_quant_cfg = get_quantizer(quant_cfg, model=model)
        # model, replace_layers_later = quantizer.get_quantized_model(model_quant_cfg, inplace=True)
        # quantize last layers (batchnorm). parmams last saved checkpoint do not entirely reflect current model anymore
        # if replace_layers_later is not None:
        #    model, _ = quantizer.get_quantized_model(replace_layers_later)
    assert isinstance(model_wrapper,
                      DynamicCheckpointQTRModelWrapper), f'Model should be torch module, not {type(model_wrapper)}'
    # only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    # this is a problem if a layer uses a non-named Tensor during the forward pass
    model_wrapper.to(device=device)
    if hasattr(log, "trace"): log.trace(model_wrapper.model)


    log.info(f"Starting benchmark")
    # for now. This just prevent the error msg, maybe in the future we find a way of using the hf-tok-parallelism feature
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_singleton.tokenizer = cfg.tokenizer
    from qtransform.dataset import DataLoaderWrapper, DatasetSplitType
    dataloader_wrapper = DataLoaderWrapper(cfg.dataset)
    train_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.TRAIN)
    eval_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.EVAL)


    bench(cfg, model_wrapper, eval_dataloader)


def bench(cfg, model_wrapper: QTRModelWrapper, dataloader: DataLoader) -> None:
    monitor = ZeusMonitor()

    idle_time = cfg.run.idle_time
    idle_measurement = measure_idle_energy(idle_time, monitor)

    generation_measurements = measure_generation_energy(cfg, model_wrapper, dataloader, monitor)

    df_verbose = pd.DataFrame(columns=['time(s)', 'cpu_energy(J)', 'gpu_energy(J)'])

    avg_cpu_energy_idle = sum(idle_measurement.cpu_energy.values()) / idle_time if idle_time > 0 else 0
    avg_gpu_energy_idle = sum(idle_measurement.gpu_energy.values()) / idle_time if idle_time > 0 else 0
    for measurement in generation_measurements:
        df_verbose.loc[len(df_verbose)] = {
            'time(s)': measurement.time,
            'cpu_energy(J)': sum(
                measurement.cpu_energy.values()) - avg_cpu_energy_idle * measurement.time,
            'gpu_energy(J)': sum(
                measurement.gpu_energy.values()) - avg_gpu_energy_idle * measurement.time}

    total_time = df_verbose['time(s)'].sum()
    total_cpu_energy = df_verbose['cpu_energy(J)'].sum()
    total_gpu_energy = df_verbose['gpu_energy(J)'].sum()

    avg_time = df_verbose['time(s)'].mean()
    avg_cpu_energy = df_verbose['cpu_energy(J)'].mean()
    avg_gpu_energy = df_verbose['gpu_energy(J)'].mean()

    df_summary = pd.DataFrame({
        'total_time(s)': [total_time],
        'avg_time(s)': [avg_time],
        'total_cpu_energy(J)': [total_cpu_energy],
        'avg_cpu_energy(J)': [avg_cpu_energy],
        'total_gpu_energy(J)': [total_gpu_energy],
        'avg_gpu_energy(J)': [avg_gpu_energy]
    })

    save_results(cfg, df_verbose, df_summary)


def measure_idle_energy(idle_time: int, monitor: ZeusMonitor) -> Measurement:
    measurement = zeus.monitor.energy.Measurement(0, {0: 0}, {0: 0}, {0: 0})
    if idle_time > 0:
        log.info(f"Measuring energy while idle. Set idle time is {idle_time} seconds")
        monitor.begin_window("Idle")
        time.sleep(idle_time)
        measurement = monitor.end_window("Idle")
        log.info(f"Measuring idle energy complete.")
    return measurement


def measure_generation_energy(cfg, model_wrapper: QTRModelWrapper, dataloader: DataLoader, monitor: ZeusMonitor) -> \
list[Measurement]:
    measurements = []
    lens = min(len(dataloader), cfg.run.max_iters)
    if isinstance(model_wrapper.model, torch.nn.Module):
        model_wrapper.model.eval()
    log.info("Measuring energy consumption during generation")
    for i, data in enumerate(dataloader):

        log.debug(f'Iteration: {i}')
        if i >= lens:
            break
        inputs = None
        if len(data) > 2:
            inputs = data['input_ids']
        elif len(data) == 2:
            inputs, _ = data
        else:
            log.error(f"unsupported dataloader output. len was {len(data)}. ")
            raise NotImplementedError
        with torch.no_grad():
            inputs = inputs.to(device_singleton.device)

            monitor.begin_window("Generation")
            y: torch.Tensor = generate(model_wrapper=model_wrapper, idx=inputs,
                                       max_new_tokens=cfg.run.max_new_tokens,
                                       temperature=cfg.run.temperature,
                                       top_k=cfg.run.top_k)
            measurement = monitor.end_window("Generation", sync_execution=True)
            measurements.append(measurement)

            # print(tokenizer.decode(y[0].tolist()) + '\n---------------\n')

    log.info("Measuring finished")
    return measurements


def save_results(cfg, df_verbose: DataFrame, df_summary: DataFrame) -> None:
    out_dir = cfg.run.out_dir
    if out_dir is not None and len(out_dir) > 0:

        if not isabs(out_dir):
            try:
                base_out_path = join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, out_dir)
            except Exception:
                base_out_path = join(getcwd(), out_dir)
        else:
            base_out_path = out_dir

        base_out_path = base_out_path.replace('~', expanduser('~'))
        if not exists(base_out_path):
            log.debug(f'Creating base energy dir: {base_out_path}')
            makedirs(base_out_path, exist_ok=True)
            # create file which keeps track of how many runs were done
            with open(join(base_out_path, 'runs.txt'), 'x') as f:
                f.write("1")

        # read and increment run number
        with open(join(base_out_path, 'runs.txt'), "r+") as f:
            lines = f.readlines()
            run_num = int(lines[0])
            f.seek(0)
            f.writelines(str(run_num + 1))

        run_folder = join(base_out_path, str(run_num))
        if not exists(run_folder):
            log.debug(f'Creating run dir: {run_folder}')
            makedirs(run_folder, exist_ok=True)
            with open(join(run_folder, 'run_cfg.txt'), 'x') as f:
                f.write(str(cfg.run))

            out_path_verbose = join(run_folder, "energy_verbose.csv")
            out_path_averages = join(run_folder, "energy_averages.csv")

            log.info(f'Saving results to: "{run_folder}"')
            df_verbose.to_csv(out_path_verbose, sep=";", index=False, mode="w", header=True)
            df_summary.to_csv(out_path_averages, sep=";", index=False, mode="w", header=True)
