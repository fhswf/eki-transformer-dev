import subprocess
import time

base_path = f"/home/student/energy"

data_points = 1000

model_checkpoints = {
    "energy_bench": [
        #{"checkpoint": f"{base_path}/chkpts/gpt2-s256-t2048-l1-h2-e256-AReLU-NBatchNormTranspose-Plearned_fhswf__TinyStoriesV2_cleaned_2024-06-13_10_56_37__epoch_1", "max_new_tokens": 64, "path" : f"{base_path+'/cp_gpt2-s256-t2048-l1-h2-e256_gte_1s'}", "cstring" : "MGPT-s256-t2048-l1-h2-e256-AReLU-NBatchNormTranspose-Plearned"},
        #{"checkpoint": f"{base_path}/chkpts/gpt2-s256-t2048-l2-h4-e256-AReLU-NBatchNormTranspose-Plearned_fhswf__TinyStoriesV2_cleaned_2024-06-13_10_53_07__epoch_1", "max_new_tokens": 32, "path" : f"{base_path+'/cp_gpt2-s256-t2048-l2-h4-e256_gte_1s'}", "cstring" : "MGPT-s256-t2048-l2-h4-e256-AReLU-NBatchNormTranspose-Plearned"},
        #{"checkpoint": f"{base_path}/chkpts/gpt2-s512-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned_fhswf__TinyStoriesV2_cleaned_2024-06-12_14_44_22__epoch_1", "max_new_tokens": 4, "path" : f"{base_path+'/cp_gpt2-s512-t2048-l2-h4-e512_gte_1s'}", "cstring" : "MGPT-s512-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned"},

        {"checkpoint": None, "max_new_tokens": 256, "path" : f"{base_path+'/gpt2-s256-t2048-l1-h2-e256'}", "cstring" : "MGPT-s256-t2048-l1-h2-e256-AReLU-NBatchNormTranspose-Plearned"},
        {"checkpoint": None, "max_new_tokens": 256, "path" : f"{base_path+'/gpt2-s256-t2048-l2-h4-e256'}", "cstring" : "MGPT-s256-t2048-l2-h4-e256-AReLU-NBatchNormTranspose-Plearned"},
        {"checkpoint": None, "max_new_tokens": 256, "path" : f"{base_path+'/gpt2-s512-t2048-l2-h4-e512'}", "cstring" : "MGPT-s512-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned"},

        #{"checkpoint": f"{base_path}/chkpts/gpt2-s256-t2048-l2-h4-e256-AReLU-NBatchNormTranspose-Plearned_fhswf__TinyStoriesV2_cleaned_2024-06-13_10_53_07__epoch_1", "max_new_tokens": 64, "path" : f"{base_path+'/cp_gpt2-s256-t2048-l2-h4-e256_same_token'}", "cstring" : "MGPT-s256-t2048-l2-h4-e256-AReLU-NBatchNormTranspose-Plearned"},
        #{"checkpoint": f"{base_path}/chkpts/gpt2-s512-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned_fhswf__TinyStoriesV2_cleaned_2024-06-12_14_44_22__epoch_1", "max_new_tokens": 64, "path" : f"{base_path+'/cp_gpt2-s512-t2048-l2-h4-e512_same_token'}", "cstring" : "MGPT-s512-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned"},
    ]
}

base_params = ["qtransform", "run=energy", "dataset=tsV2", "tokenizer=tsV2", "wandb.enabled=False",
               f"run.data_points={data_points}", "run.preheat.data_points=5", "run.idle_time=60"]

params = []

# temps = [0.7, 1.0, 1.3]
# top_ks = [1, 100, 200]
batch_sizes = [1, 2, 4, 8, 12]
temps = [0.7]
top_ks = [200]

for batch_size in batch_sizes:
    for temp in temps:
        for top_k in top_ks:
            params.append([f"run.temperature={temp}", f"run.top_k={top_k}", f"run.batch_size={batch_size}"])

commands = list()

for model, checkpoints in model_checkpoints.items():
    for checkpoint in checkpoints:
        for param in params:
            command = base_params.copy()
            command.extend(param)
            check = checkpoint["checkpoint"]
            max_new_tokens = checkpoint["max_new_tokens"]
            path = checkpoint["path"]
            cstring = checkpoint["cstring"]
            command.extend([f"model={model}", f"run.max_new_tokens={max_new_tokens}", f"run.out.path={path}",
                            f"model.cstr={cstring}"])
            if check is not None:
                command.append(f"model.checkpoint={check}")
            commands.append(command)

for command in commands:
    subprocess.run(command)
    time.sleep(60)
