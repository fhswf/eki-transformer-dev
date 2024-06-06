
# from datasets import load_dataset
# dataset = load_dataset("roneneldan/TinyStories")
# print(dataset)
# print(dataset['train'][0])
# 
# import pyarrow.parquet as pq
a = "/home/kuhmichel/.qtransform/datasets/huggingface/roneneldan/TinyStories/cache-EleutherAI/gpt-neo-125M-128-EVAL-grouped.arrow"
b = "/home/kuhmichel/Downloads/0000(1).parquet"
c = "hf://datasets/roneneldan/TinyStories@691b0d9bd48ade766778c940011ca1c549f6359b/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
d = "/home/kuhmichel/.cache/huggingface/datasets/roneneldan___tiny_stories/default-ca0b42f82bf0eeae/0.0.0/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02/tiny_stories-train-00000-of-00004.arrow"
# print(pq.read_metadata(b))
# table = pq.read_table(b)
# print(table)
# df = table.to_pandas()
# print(df.describe())
# 
# for e in df["text"]:
#     print(e)
#     break

"".splitlines
from datasets import Dataset
ds = Dataset.from_parquet(b)
print(ds)
print(ds['text'][0])
ds = Dataset.from_file(d)
print(ds)
print(ds['text'][0])

# import pyarrow as pa
# with pa.memory_map(d, 'r') as source:
#     loaded_arrays = pa.ipc.open_file(source).read_all()
# arr = loaded_arrays[0]
# print(f"{arr[0]} .. {arr[-1]}")