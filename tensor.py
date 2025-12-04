import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class TRTWrapper:
    """
    Handles building the TensorRT engine and running inference.
    Manages GPU memory allocation via PyCUDA.
    """
    def __init__(self, engine_path, onnx_path=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.engine = None
        self.context = None

    def build_engine(self):
        """Builds a TensorRT engine from the ONNX file."""
        if not self.onnx_path:
            raise ValueError("ONNX path required to build engine")
            
        print(f"Building TensorRT Engine from {self.onnx_path}...")
        
        # Setup Builder
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.logger)

        # Enable FP16 (Half Precision)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 Mode Enabled!")
        
        # Memory Config (1GB Workspace)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Parse ONNX
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse_from_file(self.onnx_path):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return


        profile = builder.create_optimization_profile()
        # Input name 'input' matches model.py export
        # Args: name, min_shape, opt_shape, max_shape
        # optimized for batch size 1, but allow up to 64
        profile.set_shape("input", (1, 3, 32, 32), (1, 3, 32, 32), (64, 3, 32, 32))
        config.add_optimization_profile(profile)

        try:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("Error: Builder failed to produce an engine.")
                return

            with open(self.engine_path, "wb") as f:
                f.write(serialized_engine)
            print(f"Engine built and saved to {self.engine_path}")
        
        except Exception as e:
            print(f"Build failed: {e}")

    def load_engine(self):
        """Loads the serialized engine from disk."""
        if not os.path.exists(self.engine_path):
            self.build_engine()
        
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if not self.engine:
                raise RuntimeError("Failed to deserialize engine.")

            self.context = self.engine.create_execution_context()
            
            # specify input shape because it's dynamic
            self.context.set_input_shape("input", (1, 3, 32, 32))

    def infer(self, input_data):
        """
        Runs inference on the engine.
        input_data: numpy array (1, 3, 32, 32)
        """
        if not self.context:
            self.load_engine()

        # Flatten Input
        input_data = input_data.ravel().astype(np.float32)
        
        # Allocate Device Memory (PyCUDA)
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(10 * 4) # 10 classes, float32
        
        #CUDA Stream
        stream = cuda.Stream()

        #Host -> Device
        cuda.memcpy_htod_async(d_input, input_data, stream)
        
        self.context.set_tensor_address("input", int(d_input))
        self.context.set_tensor_address("output", int(d_output))
        
        self.context.execute_async_v3(stream_handle=stream.handle)
        
        # Device -> Host
        h_output = np.empty(10, dtype=np.float32)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
        stream.synchronize()
        
        return h_output