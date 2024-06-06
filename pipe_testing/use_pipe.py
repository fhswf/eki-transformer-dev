import hydra
from omegaconf import DictConfig, OmegaConf
from logging import getLogger
log = getLogger(__name__)

"""
Test if usage of pipes is a good way to communicate between python processes
"""
@hydra.main(version_base=None, config_path="conf", config_name="cfg")
def my_app(cfg : DictConfig) -> None:
    #log.info(f'Using config: "{cfg}')
    if cfg.get('pipe') is None:
        log.error(f'pipe not specified')
    if cfg.get('run') is None:
        log.error(f'run not specified')
    
    match cfg.run:
        case 'create':
            #should access mode be w+ or a?
            #writing into a pipe needs to happen in background
            #maybe append some generic args before text in order to differentiate between outputs in pipe
            with open(cfg.pipe, 'w') as pipe:
                text = str(cfg.args) + ":" + cfg.text
                log.info(f'Writing text: "{cfg.text}" into pipe: {text}')
                pipe.write(text)
        case 'consume': 
            #maybe buffering=0
            #input redirection with use_pipe.py < pipe.fifo could also be useful
            with open(cfg.pipe, 'r') as pipe:
                text = pipe.readline()
                log.info(f'Read text: "{text}" from pipe {cfg.pipe}')
if __name__ == "__main__":
    my_app()
