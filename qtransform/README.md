# qtransform
Quantization for transformer models with brevitas


## How to run
Install via pip e.g. :
```
pip install git+https://github.com/fhswf/eki-transformer-dev.git@develop#subdirectory=qtransform 
```
Run the module:
```
qtransform run=train model=<ModelConfig.yaml> dataset=<DatasetType> run.epochs=1 run.max_iters=300 dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2
```
Our module uses [hydra](https://hydra.cc/docs/intro/). Commandline options and generated configs for the run can be viewed with --help
```
qtransform run=train --help
```
Note that `run=<OPTION>` determines what the programm should run. If you leave this empty or miss any other parameter the cli will generally tell you what the options are available. Note that bash completion does not work currently on all systems.

### Extra params
Sometimes extra cli params can be injected that are not part of the global config.
Here is a list:
``` bash
# for resume training or loading a saved model (run=train,export,bench)
+from_checkpoint=<path_to_torchmodel.pt>
# custom log level for development
+trace=True
# for run=train, perform export after all epochs are done
+export=True
```

### Examples
```
qtransform run=train model=gpt_2_h2l2e256b64_GeBN dataset=huggingface debug=True dataset.name=openwebtext +export=True run.epochs=100 run.ma
x_iters=300 dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2
```
