import argparse
import numpy as np
import torch
import os
import time
import model
import tensor

def get_dummy_input():
    # Batch size 1, 3 Channels, 32x32 image
    return np.random.random((1, 3, 32, 32)).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="AI Pipeline Manager")
    
    # Actions
    parser.add_argument('action', choices=['train', 'export', 'torch', 'tensorrt'], 
                        help="Action to perform")

    # Parameters
    parser.add_argument('--model_path', type=str, default='simple_cnn.pth', 
                        help="Path for .pth file")
    parser.add_argument('--onnx_path', type=str, default='simple_cnn.onnx', 
                        help="Path for .onnx file")
    parser.add_argument('--engine_path', type=str, default='simple_cnn.engine', 
                        help="Path for .engine file")
    parser.add_argument('--epochs', type=int, default=20, 
                        help="Number of training epochs")

    args = parser.parse_args()

    # --- TRAIN ---
    if args.action == 'train':
        print(f"Starting training for {args.epochs} epochs...")
        start_time = time.time()
        
        trained_model = model.train_model(epochs=args.epochs)
        model.save_model(trained_model, args.model_path)
        
        print(f"Total Training Time: {time.time() - start_time:.2f}s")

    # --- EXPORT ---
    elif args.action == 'export':
        try:
            loaded_model = model.load_model(args.model_path)
            model.export_to_onnx(loaded_model, args.onnx_path)
        except Exception as e:
            print(f"Export failed: {e}")

    # --- TORCH INFERENCE ---
    elif args.action == 'torch':
        print(f"Running Torch Inference using {args.model_path}")
        try:
            loaded_model = model.load_model(args.model_path)
            np_input = get_dummy_input()
            torch_input = torch.from_numpy(np_input)
            
            # Warmup (optional but recommended for fair comparison)
            print("Warming up GPU...")
            for _ in range(10):
                model.run_torch_inference(loaded_model, torch_input)

            # Benchmark
            print("Benchmarking...")
            start_time = time.time()
            result = model.run_torch_inference(loaded_model, torch_input)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            print(f"Predicted Class: {np.argmax(result)}")
            print(f"PyTorch Latency: {latency:.4f} ms")
            
        except Exception as e:
            print(f"Torch inference failed: {e}")

    # --- TENSORRT INFERENCE ---
    elif args.action == 'tensorrt':
        print(f"Running TensorRT Inference...")
        
        if not os.path.exists(args.onnx_path):
            print(f"Warning: {args.onnx_path} not found. Please run 'export' first.")
            return

        trt_wrapper = tensor.TRTWrapper(args.engine_path, args.onnx_path)
        
        try:
            np_input = get_dummy_input()

            print("Building/Loading Engine & Warming up...")
            trt_wrapper.infer(np_input)
            
            # Benchmark
            print("Benchmarking...")
            start_time = time.time()
            result = trt_wrapper.infer(np_input)
            end_time = time.time()

            latency = (end_time - start_time) * 1000
            print(f"Predicted Class: {np.argmax(result)}")
            print(f"TensorRT Latency: {latency:.4f} ms")

        except Exception as e:
            print(f"TensorRT inference failed: {e}")

if __name__ == "__main__":
    main()