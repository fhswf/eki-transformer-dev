import torch , brevitas
t = torch.arange(25, dtype=torch.float32).reshape((5,5))
print(t)

mask = torch.ones((5,5), dtype=torch.float32).tril(diagonal=0)
mask = mask.masked_fill(mask == 0, float('-inf'))
print(mask)
mask_bool_trans = torch.ones((5,5), dtype=bool).tril(diagonal=-1).transpose(-1,-2)
print(mask_bool_trans)
mask_bool = torch.ones((5,5), dtype=bool).tril(diagonal=0)
print(mask_bool)

print("sdp outs")
print(torch.nn.functional.scaled_dot_product_attention(t, t, t, attn_mask=mask))
print(torch.nn.functional.scaled_dot_product_attention(t, t, t, attn_mask=mask_bool))
print(torch.nn.functional.scaled_dot_product_attention(t, t, t, attn_mask=mask_bool_trans)) # this does not work
print(torch.nn.functional.scaled_dot_product_attention(t, t, t,is_causal=True)) 
from brevitas import nn as qnn

MHA = torch.nn.MultiheadAttention(5, 1, batch_first=True)
QMHA = qnn.QuantMultiheadAttention(5, 1, batch_first=True, packed_in_proj=False)

print("MHA outs")
print(MHA(t,t,t,  attn_mask = mask))
print(MHA(t,t,t,  attn_mask = mask_bool)) # produces nans
print(MHA(t,t,t,  attn_mask = mask_bool_trans))
print("with is causal")
# for newer torch version
#print(MHA(t,t,t, is_causal=True, attn_mask = mask))
#print(MHA(t,t,t, is_causal=True, attn_mask = mask_bool)) # produces nans
#print(MHA(t,t,t, is_causal=True, attn_mask = mask_bool_trans))
# for torch < 2.1
print(MHA(t,t,t, is_causal=True))
print(MHA(t,t,t, is_causal=True)) # produces not  nans ! output matches the above
print(MHA(t,t,t, is_causal=True))

print("QMHA outs")
print(QMHA(t,t,t,  attn_mask = mask))
print(QMHA(t,t,t,  attn_mask = mask_bool))  # produces nans
print(QMHA(t,t,t,  attn_mask = mask_bool_trans))
