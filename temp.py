from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.export import export_onnx_qop
import torch
import brevitas.nn as qnn

IN_CH = 3
IMG_SIZE = 128
OUT_CH = 128
BATCH_SIZE = 1

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantConv2d(
            IN_CH, OUT_CH, kernel_size=3, bias=True,
            weight_bit_width=4, bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat)

    def forward(self, inp):
        inp = self.input_quant(inp)
        inp = self.linear(inp)
        return inp

inp = torch.randn(BATCH_SIZE, IN_CH, IMG_SIZE, IMG_SIZE)
model = Model()
model.eval()


export_onnx_qop(
    model, args=inp, export_path="quant_model_qop.onnx")