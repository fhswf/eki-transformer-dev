import subprocess

base_path = "/home/student/energy_measurements/"

max_new_tokens = 64

max_iters = 100

model_checkpoint = {
    "EB_gpt_small_l1_h2": "/home/student/energy_measurements/chkpts/2025-04-30_12:20:10-quaint-speakerEB_gpt_small_l1_h2tsV2",
    #"EB_gpt_small_l2_h4": None,
    #"EB_gpt_medium_l1_h2": None,
    #"EB_gpt_medium_l2_h4": None,
    #"EB_gpt_small_l1_h2": "",
    #"EB_gpt_small_l2_h4": "",
    #"EB_gpt_medium_l1_h2": "",
    #"EB_gpt_medium_l2_h4": "",
}

base_params = ["qtransform", "run=energy", "dataset=tsV2", "tokenizer=tsV2", "wandb.enabled=False",
               f"run.max_iters={max_iters}", f"run.max_new_tokens={max_new_tokens}"]

params = []

params.append(["run.preheat.max_iters=100", "run.preheat.max_new_tokens=64", f"run.out.path={base_path+'preheat'}"])

temps = [0.7, 1.0, 1.3]
top_ks = [1, 100, 200]

for temp in temps:
    for top_k in top_ks:
        params.append([f"run.temperature={temp}", f"run.top_k={top_k}", f"run.out.path={base_path+'top_k_temps'}"])

commands = list()

for model, checkpoint in model_checkpoint.items():
    for param in params:
        command = base_params.copy()
        command.extend(param)
        command.append(f"model={model}")
        if checkpoint is not None:
            command.append(f"model.checkpoint={checkpoint}")
        commands.append(command)

for command in commands:
    subprocess.run(command)
