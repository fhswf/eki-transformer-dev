import onnxruntime as ort
_default_session_options = ort.capi._pybind_state.get_default_session_options()

def get_default_session_options_new():
    """Return specific default session options for onnxruntime."""
    _default_session_options.inter_op_num_threads = 1
    _default_session_options.intra_op_num_threads = 1
    return _default_session_options

ort.capi._pybind_state.get_default_session_options = get_default_session_options_new


# Also make QONNX related passes available when importing this file, these are
# non-default passes
import onnx_passes.passes.imports.qonnx  # noqa: Passes used via registry
import onnx_passes.passes.inline.qonnx  # noqa: Passes used via registry

# IThe passes module sets up the registry and makes the @passes.register
# decorator work
import onnx_passes.passes as passes


# Custom cleanup pass to wrap a sequence of cleanup and annotation passes in an
# exhaustive manner
@passes.verify.tolerance
@passes.register("tidyup")
class Tidyup(passes.base.Transformation, passes.compose.ComposePass):
    __passes__ = [
        "shape-inference",
        "fold-constants",
        "eliminate",
        "cleanup",
    ]

    __exhaustive__ = True

