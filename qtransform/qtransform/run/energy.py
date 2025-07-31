import logging
from os.path import join, expanduser, exists
from os import makedirs
import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame
from zeus.monitor import ZeusMonitor, Measurement
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform.model import QTRModelWrapper
import torch
from datetime import datetime
import time

from qtransform.run.infer import generate
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

    torch.manual_seed(cfg.seed)
    tokenizer_singleton.tokenizer = cfg.tokenizer
    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:
        if not model_wrapper.quantized:
            log.info(f'Quantizing model')
            model_wrapper.quantize_model(quant_cfg)
        else:
            warn_once(log, f'Model was already quantized, ignoring quant_cfg from hydra')

    # only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    # this is a problem if a layer uses a non-named Tensor during the forward pass
    model_wrapper.to(device=device_singleton.device)
    if hasattr(log, "trace"): log.trace(model_wrapper.model)

    log.info(f"Starting benchmark")

    bench(cfg, model_wrapper)


def bench(cfg, model_wrapper: QTRModelWrapper) -> None:
    """
    Driver code for energy benchmark
    """
    monitor = ZeusMonitor()

    idle_time = cfg.run.idle_time
    idle_measurements = measure_idle_energy(idle_time, monitor)

    preheat_measurements = measure_generation_energy(cfg, model_wrapper, monitor, preheating=True)

    generation_measurements = measure_generation_energy(cfg, model_wrapper, monitor, preheating=False)

    df_verbose = pd.DataFrame(columns=['time(s)', 'cpu_energy(J)', 'gpu_energy(J)', 'start', 'end', 'tokens', 'type'])

    for measurements in [idle_measurements, preheat_measurements, generation_measurements]:
        for _measurement in measurements:
            measurement = _measurement['measurement']
            start = _measurement['start']
            end = _measurement['end']
            tokens = _measurement.get('tokens')
            type_str = _measurement['type']
            if not measurement.cpu_energy:
                # If Zeus can't find any CPU, the cpu_energy dict is none, so here it's manually set to 0
                measurement.cpu_energy = {0: 0}
            df_verbose.loc[len(df_verbose)] = {
                'time(s)': measurement.time,
                'cpu_energy(J)': sum(
                    measurement.cpu_energy.values()),
                'gpu_energy(J)': sum(
                    measurement.gpu_energy.values()),
                'start': start,
                'end': end,
                'tokens': tokens,
                'type': type_str,
            }

    save_results(cfg, df_verbose)


def measure_idle_energy(idle_time: int, monitor: ZeusMonitor) -> list[dict[str, Measurement | float]]:
    """
    Measures energy while idling. Intended to be used before doing ANY generation. Useful for visualizing energy consumption before generation.
    """
    measurements = []
    if idle_time > 0:
        log.info(f"Measuring energy while idle. Set idle time is {idle_time} seconds")
        for i in range(idle_time):
            begin = time.time()
            monitor.begin_window("Idle")
            time.sleep(1)
            measurement = monitor.end_window("Idle")
            end = time.time()
            measurements.append({"measurement": measurement, "start": begin, "end": end, "type": "idle"})
        log.info(f"Measuring idle energy complete.")
    return measurements


def measure_generation_energy(cfg, model_wrapper: QTRModelWrapper, monitor: ZeusMonitor,
                              preheating: bool) -> list[dict[str, Measurement | float]]:
    """
    Mesures energy during generation.
    max_new_tokens, data_points, temperature and top_k can be set through the configs run parameters.
    If preheating, the data_points from the preheating section are used instead.
    """
    measurements = []
    log_msg_start = "Measuring energy consumption during generation"
    log_msg_end = "Measuring finished"
    type_str = "generation"
    if preheating:
        log_msg_start = "Starting preheating"
        log_msg_end = "Preheating finished"
        type_str = "preheat"
        lens = cfg.run.preheat.data_points
    else:
        lens = cfg.run.data_points

    if lens > 0:
        if isinstance(model_wrapper.model, torch.nn.Module):
            model_wrapper.model.eval()
        log.info(log_msg_start)

        with torch.no_grad():
            for i in range(0, lens, cfg.run.batch_size):
                current_batch_size = min(cfg.run.batch_size, lens - i)
                inputs = torch.randint(0, 2048, [current_batch_size, 512], device=device_singleton.device)

                begin = time.time()
                monitor.begin_window("Generation")
                outputs: torch.Tensor = generate(model_wrapper=model_wrapper, idx=inputs,
                                                 max_new_tokens=cfg.run.max_new_tokens,
                                                 temperature=cfg.run.temperature,
                                                 top_k=cfg.run.top_k)
                measurement = monitor.end_window("Generation", sync_execution=True)
                end = time.time()
                measurements.append({
                    "measurement": measurement, "start": begin, "end": end, "type": type_str,
                    "tokens": current_batch_size * cfg.run.max_new_tokens,
                })

        log.info(log_msg_end)
    return measurements


def save_results(cfg, df_verbose: DataFrame) -> None:
    """
    Saves results. Results of measurements are stored alongside the configs run parameters in a folder
    with the specific run number. The run number starts at 1 and is stored between different calls of the
    qtransform energy command and is incremented by each call using this function.

    If no path is specified, the results will be printed instead.

    Intended to be used after measuring energy during generation.
    """
    base_out_path = cfg.run.out.path
    if base_out_path:

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

            log.info(f'Saving results to: "{run_folder}"')
            df_verbose.to_csv(out_path_verbose, sep=";", index=False, mode="w", header=True)

    else:
        print('\n---------------\n')
        print(df_verbose)
        print('\n---------------\n')
