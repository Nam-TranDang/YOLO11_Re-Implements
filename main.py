# Not Touch Yet - Unused file 27/Oct/2026
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Text Detection Model Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train --config config/ocr_config.yaml
  
  # Evaluate model
  python main.py evaluate --config config/ocr_config.yaml --model checkpoints/best_model.pth
  
  # Run inference
  python main.py predict --config config/ocr_config.yaml --image data/test_image.jpg
  
  # Export model
  python main.py export --config config/ocr_config.yaml --model checkpoints/best_model.pth --format onnx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train OCR model')
    train_parser.add_argument('--config', required=True, help='Config file path')
    train_parser.add_argument('--resume', help='Resume from checkpoint')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate OCR model')
    eval_parser.add_argument('--config', required=True, help='Config file path')
    eval_parser.add_argument('--model', required=True, help='Model checkpoint path')
    eval_parser.add_argument('--output', help='Output metrics file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run OCR inference')
    predict_parser.add_argument('--config', required=True, help='Config file path')
    predict_parser.add_argument('--image', required=True, help='Input image path')
    predict_parser.add_argument('--output', help='Output JSON file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--config', required=True, help='Config file path')
    export_parser.add_argument('--model', required=True, help='Model checkpoint path')
    export_parser.add_argument('--format', choices=['onnx', 'hf'], required=True, help='Export format')
    export_parser.add_argument('--output', required=True, help='Output directory')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--config', required=True, help='Config file path')
    preprocess_parser.add_argument('--input_dir', required=True, help='Input directory')
    preprocess_parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import and run appropriate command
    if args.command == 'train':
        from cli.train import main as train_main
        sys.argv = ['train.py', '--config', args.config]
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.seed:
            sys.argv.extend(['--seed', str(args.seed)])
        train_main()
        
    elif args.command == 'evaluate':
        from cli.evaluate import main as eval_main
        sys.argv = ['evaluate.py', '--config', args.config, '--model', args.model]
        if args.output:
            sys.argv.extend(['--output', args.output])
        eval_main()
        
    elif args.command == 'predict':
        from inference.ocr_pipeline import main as predict_main
        # Construct arguments for prediction
        sys.argv = ['ocr_pipeline.py', 
                   '--config', args.config,
                   '--detection_model', args.model,
                   '--recognition_model', args.model,
                   '--image', args.image]
        if args.output:
            sys.argv.extend(['--output', args.output])
        predict_main()
        
    elif args.command == 'export':
        if args.format == 'onnx':
            from export.export_onnx import main as export_main
            sys.argv = ['export_onnx.py', '--config', args.config, 
                       '--model', args.model, '--output_dir', args.output]
        elif args.format == 'hf':
            from export.export_hf_format import main as export_main
            sys.argv = ['export_hf_format.py', '--config', args.config,
                       '--model', args.model, '--output', args.output]
        export_main()
        
    elif args.command == 'preprocess':
        from preprocessing.document_preprocess import main as preprocess_main
        sys.argv = ['document_preprocess.py', '--config', args.config,
                   '--input_dir', args.input_dir, '--output_dir', args.output_dir]
        preprocess_main()

if __name__ == '__main__':
    main()
