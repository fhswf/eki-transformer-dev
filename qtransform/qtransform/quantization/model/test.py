from brevitas import nn as qnn 
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

# maybe be loading some pyython file instead of yaml we can save some effort?
c = {
    "mha" : (qnn.QuantMultiheadAttention, {"output_quant":Int8ActPerTensorFloat, "return_quant_tensor":True})
}