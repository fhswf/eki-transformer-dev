import yaml
import pandas as pd
import numpy as np
import os

import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.range_analysis import RangeInfo
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

# Custom build steps required to streamline and convert the attention operator
from finn.builder.custom_step_library.transformer import (
    prepare_graph,
    set_fifo_depths,
    set_target_parallelization,
    step_apply_folding_config,
    step_convert_attention_to_hw,
    step_convert_depth_wise_to_hw,
    step_convert_elementwise_binary_to_hw,
    step_convert_lookup_to_hw,
    step_convert_split_concat_to_hw,
    step_replicate_streams,
    step_streamline,
)

if __name__ == "__main__":

    # Extract sequence length and embedding dimension from the verification
    model_name = os.path.basename(os.environ["CALL_MODEL_NAME"])
    # check if input tensor has more then 2 dimensions
    in_shape = np.load(model_name + ".inp.npy").shape
    print("Input shape: " + str(in_shape))
    # here we assume embbeddings to be computed in the model graph, first dim is alway batch size
    if len(in_shape) == 2:
        batch_dim, seq_len = in_shape
        # try and extract emb_dim from model graph. emb should de the last dim of the first layer in the model
        # look for the first node in the graph of type "Gather" and get its last dim:
        try:
            model = ModelWrapper(onnx.load(model_name))
            gather_nodes = model.get_nodes_by_op_type("Gather")
            first_gather_node = gather_nodes[0]
            emb_dim = model.get_tensor_shape(first_gather_node.input[1])[-1]
            print("Extracted emb_dim from model graph: " + str(emb_dim))
        except:
            emb_dim = 256
            print("Could not extract emb_dim from model graph, using default: " + str(emb_dim))
    else:
        batch_dim, seq_len, emb_dim = in_shape
    
    # we can only handle batch size 1 for now.
    # convert all inputs in onnx graph to batch size 1
    if batch_dim != 1:
        print("Original batch size: " + str(batch_dim))
        print("Setting batch size to 1 in model graph...")
        # TODO
        print("Could not change batch size to 1 in model graph, error: " + str(e))
        exit(1)
    # Read the input value range information for the dataset from the parameters
    # Note: Consider calibrating this on the fly from the dataset
    input_range = tuple(np.array([ -100, +100 ]).T)
    # Construct the seed range information of the input tensor
    if len(in_shape) == 2:
        range_info = RangeInfo(shape=(1, seq_len), range=input_range)
    else:
        range_info = RangeInfo(shape=(1, seq_len, emb_dim), range=input_range)
    print("Using range_info: " + str(range_info))
    cfg = build_cfg.DataflowBuildConfig(
        verbose=True,
        folding_config_file="folding.yaml",
        specialize_layers_config_file="specialize_layers.json",
        standalone_thresholds=True,
        max_multithreshold_bit_width=16,
        mvau_wwidth_max=2048,
        # not sure if we need to set this higher or weere the limit is
        target_fps= 100,
        output_dir=os.environ["FINN_BUILD_DIR"],
        board="U55C",  # U280, U55C, 
        shell_flow_type="vitis_alveo",
        # use finn branch feature multi-fpga support
        
        # # Multi-FPGA specific configuration options
        # num_fpgas: int = 2
        # # The number of ports per device - this might change in meaning,
        # # depending on the communication kernel used
        # ports_per_device: int = 2
        # # What strategy to use to partition the dataflow graph
        # partition_strategy: PartitioningStrategy = PartitioningStrategy.RESOURCE_UTILIZATION
        # # Tells the flow what topology to use. This determines the transformation
        # # that creates the network metadata necessary for kernel packing
        # topology: MFTopology = MFTopology.CHAIN
        # # What kind of kernel is used to communicate in the network
        # communication_kernel: MFCommunicationKernel = MFCommunicationKernel.AURORA
        # # How much a FPGA can be utilized at max. The solver will fail if it
        # # cannot comply with this limitation
        # max_utilization: float = 0.85
        # # How much resources of a single FPGA should be used ideally. Used in some objective
        # # functions.
        # ideal_utilization: float = 0.75

        # Number of synthesis workers to run in parallel    
        # Determines how many synthesis processes can run in parallel. Keep in mind
        # that very roughly estimated, one synthesis should be able to use up to 100 GB RAM,
        # (sometimes more) depending on the model and version of Vivado. For example on a
        # 512 GB node, you can run roughly 4 Synthesis in parallel.
        # Defaults to 1, in order not to crash local computers with OOM errors
        # parallel_synthesis_workers: int = 1
        
        verify_steps=[
            # In this custom flow, VerificationStepType is repurposed as follows:
            # Enables "tidied_up_python" (before lowering) verification in step_prepare_graph
            build_cfg.VerificationStepType.TIDY_UP_PYTHON,
            # Enables "lowered_python" (after lowering) and "prepared_graph_python"
            # (after QONNX to FINN conversion) verification in step_prepare_graph
            build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
            # Enables "streamlined_python" verification after custom step_streamline
            build_cfg.VerificationStepType.STREAMLINED_PYTHON,
            # Enables "folded_hls_cppsim" verification after custom step_apply_folding_config
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            # Enables RTL simulation of default steps contained in the flow:
            build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,  # after step_hw_ipgen
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,  # after step_create_stitched_ip
        ],
        
        # Build steps to execute
        steps=[
            # Prepares the QONNX graph to be consumed by FINN: Cleanup, lowering
            # and Quant to MultiThreshold conversion
            prepare_graph(range_info=range_info),
            # Unified exhaustive streamlining of complex model topologies
            # including attention, residuals and splits
            step_streamline,
            # conversion of the scaled dot-product attention pattern to
            # hardware, including cleanup and data layout squeezing
            step_convert_attention_to_hw,
            # Convert the elementwise binary operations to hardware operators.
            # These include for example adding residual branches and positional encoding
            step_convert_elementwise_binary_to_hw,
            # Convert Lookup layers, e.g., token embedding, to hardware custom operators
            step_convert_lookup_to_hw,
            # Convert Split and Concat operators to hardware, e.g., splits
            # contained in the GLU activation
            step_convert_split_concat_to_hw,
            # Convert depth-wise convolution MatMuls to VVUs
            step_convert_depth_wise_to_hw,
            # Properly replicate the stream feeding the query, key and value projections
            step_replicate_streams,
            # Convert most other layers supported by FINN to HW operators
            "step_convert_to_hw",
            # Specialize HW layer implementations as either HLS or RTL
            "step_specialize_layers",
            "step_create_dataflow_partition",
            # Set the folding configuration to meet the cycles per sequence target
            set_target_parallelization(seq_len, emb_dim),
            # Apply folding configuration, specifying hardware implementation details
            step_apply_folding_config,
            "step_minimize_bit_width",
            # The ScaledDotProductAttention custom op does not define any estimates
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",  # for synthesis results (e.g. utilization)
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
            # MutliFPA Steps
            # "step_make_multifpga"
        ],

        # Generate and keep the intermediate outputs including reports
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
 
        # File with test inputs for verification
        verify_input_npy= model_name + ".inp.npy",
        # File with expected test outputs for verification
        verify_expected_output_npy= model_name + ".onnx_out.npy",
        # Output full context dump for verification steps
        verify_save_full_context=True,
        # Save the intermediate model graphs
        save_intermediate_models=True,
        # Avoid RTL simulation for setting the FIFO sizes
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.CHARACTERIZE,
        # Do not automatically set FIFO sizes as this requires RTL simulation
        # not implemented for the attention operator
        auto_fifo_depths=False,
        # Build steps to execute
    )
    # get model name from environment variable $CALL_MODEL_NAME
    
    # Attempt to work around onnxruntime issue on Slurm-managed clusters:
    # See https://github.com/microsoft/onnxruntime/issues/8313
    # This seems to happen only when assigned CPU cores are not contiguous
    import onnxruntime as ort
    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        """Return specific default session options for onnxruntime."""
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
    
    # Run the build process on the dummy attention operator graph
    print("Starting FINN build process... " + str(model_name) + " "  + str(cfg))
    build.build_dataflow_cfg(model_name, cfg)

    if os.path.exists("finn-build/report/post_synth_resources.json"):
        # Collect and aggregate build metrics like resource utilization
        # Open the report file
        with open("finn-build/report/post_synth_resources.json") as file:
            # Load the JSON formatted report
            report = pd.read_json(file, orient="index")
        # Filter the reported rows according to some regex filter rule
        report = report.filter(
            regex="(top)", axis="rows"
        )
        # Generate a summary of the total resources
        summary = report.sum()
        # Dump the metrics dictionary as yaml
        with open("resources.yaml", "w") as file:
            # Convert the dataframe to a dictionary which can be dumped into YAML
            yaml.safe_dump(summary.to_dict(), file)
    else:     
        print("No synthesis report found, finn-build/report/post_synth_resources.json")
