import argparse
import numpy as np
import torch
import os
import time
import sys
import model
import tensor

def get_dummy_input():
    # Batch size 1, 3 Channels, 32x32 image
    return np.random.random((1, 3, 32, 32)).astype(np.float32)

def evaluate_accuracy(backend, model_wrapper, test_loader):
    """
    Runs validation on the provided test_loader and returns accuracy.
    backend: 'torch' or 'tensorrt'
    model_wrapper: the loaded PyTorch model or the TRTWrapper object
    """
    print(f"\n[Validation] Checking Accuracy on {len(test_loader)} images...")
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(test_loader):
        label = labels.numpy()[0]
        
        if backend == 'torch':
            probs = model.run_torch_inference(model_wrapper, inputs)
            prediction = np.argmax(probs)
        else:
            numpy_input = inputs.numpy() 
            probs = model_wrapper.infer(numpy_input)
            prediction = np.argmax(probs)

        if prediction == label:
            correct += 1
        total += 1
        
        if i % 1000 == 0 and i > 0:
            sys.stdout.write(f".")
            sys.stdout.flush()
            
    print("\n")
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description="AI Pipeline Manager")
    
    parser.add_argument('action', choices=['train', 'export', 'torch', 'tensorrt', 'validate'], 
                        help="Action to perform")

    # Paths
    parser.add_argument('--model_path', type=str, default='simple_cnn.pt', 
                        help="Path to save/load the PyTorch model (.pt)")
    parser.add_argument('--onnx_path', type=str, default='simple_cnn.onnx', 
                        help="Path to save/load the ONNX export")
    parser.add_argument('--engine_path', type=str, default='simple_cnn.engine', 
                        help="Path to save/load the TensorRT engine")
    
    # Configs
    parser.add_argument('--epochs', type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument('--backend', choices=['torch', 'tensorrt'], default='torch',
                        help="Backend to use for validation")

    args = parser.parse_args()


    if args.action == 'train':
        print(f"Starting training for {args.epochs} epochs...")
        start_time = time.time()
        
        trained_model = model.train_model(epochs=args.epochs)
        model.save_model(trained_model, args.model_path)
        
        print(f"Total Training Time: {time.time() - start_time:.2f}s")
        print("Next step: Run 'export' to prepare for TensorRT.")

    elif args.action == 'export':
        try:
            loaded_model = model.load_model(args.model_path)
            model.export_to_onnx(loaded_model, args.onnx_path)
            print("Next step: Run 'tensorrt' to benchmark.")
        except Exception as e:
            print(f"Export failed: {e}")

    elif args.action == 'validate':
        print(f"\n[ACTION] Standalone Validation ({args.backend.upper()})")
        
        test_loader = model.get_test_loader(batch_size=64)
        
        if args.backend == 'torch':
            loaded_model = model.load_model(args.model_path)
            start_time = time.time()
            acc = evaluate_accuracy('torch', loaded_model, test_loader)
        elif args.backend == 'tensorrt':
            if not os.path.exists(args.onnx_path):
                print("Error: ONNX file not found.")
                return
            trt_wrapper = tensor.TRTWrapper(args.engine_path, args.onnx_path)
            start_time = time.time()
            acc = evaluate_accuracy('tensorrt', trt_wrapper, test_loader)

        duration = time.time() - start_time
        print(f"Final Accuracy: {acc:.2f}%")
        print(f"Total Time: {duration:.2f}s")

    elif args.action == 'torch':
        print(f"--- Running Torch Speed & Accuracy Test ---")
        try:
            loaded_model = model.load_model(args.model_path)
            
            # 1. Speed Test (Pure Latency)
            np_input = get_dummy_input()
            torch_input = torch.from_numpy(np_input)
            
            print("\n[Speed Test] Warming up GPU...")
            for _ in range(10): model.run_torch_inference(loaded_model, torch_input)

            print("[Speed Test] Benchmarking Single-Image Latency...")
            start_time = time.time()
            result = model.run_torch_inference(loaded_model, torch_input)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            print(f"Predicted Class (Dummy): {np.argmax(result)}")
            print(f"PyTorch Pure Latency: {latency:.4f} ms")
            
            # 2. Accuracy Test
            test_loader = model.get_test_loader(batch_size=1)
            acc = evaluate_accuracy('torch', loaded_model, test_loader)
            print(f"Final Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"Torch inference failed: {e}")
            
    elif args.action == 'tensorrt':
        print(f"--- Running TensorRT Speed & Accuracy Test ---")
        if not os.path.exists(args.onnx_path):
            print(f"Warning: {args.onnx_path} not found.")
            return

        trt_wrapper = tensor.TRTWrapper(args.engine_path, args.onnx_path)
        try:
            # 1. Speed Test (Pure Latency)
            np_input = get_dummy_input()
            print("\n[Speed Test] Warming up Engine...")
            trt_wrapper.infer(np_input) 
            
            print("[Speed Test] Benchmarking Single-Image Latency...")
            start_time = time.time()
            result = trt_wrapper.infer(np_input)
            end_time = time.time()

            latency = (end_time - start_time) * 1000
            print(f"Predicted Class (Dummy): {np.argmax(result)}")
            print(f"TensorRT Pure Latency: {latency:.4f} ms")
            
            # 2. Accuracy Test
            test_loader = model.get_test_loader(batch_size=1)
            acc = evaluate_accuracy('tensorrt', trt_wrapper, test_loader)
            print(f"Final Accuracy: {acc:.2f}%")

        except Exception as e:
            print(f"TensorRT inference failed: {e}")

if __name__ == "__main__":
    main()