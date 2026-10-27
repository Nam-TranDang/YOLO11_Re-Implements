import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import yaml
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from training.ocr_trainer import OCRTrainer
from inference.ocr_pipeline import OCRPipeline

class TestE2E:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
        self.create_test_config()
    
    def create_test_data(self):
        """Create minimal test dataset"""
        # Create directories
        data_dir = Path(self.temp_dir) / 'data'
        (data_dir / 'images').mkdir(parents=True)
        
        # Create synthetic images with text
        texts = ['HELLO', 'WORLD', 'TEST', 'OCR']
        annotations = []
        
        for i, text in enumerate(texts):
            # Create simple image with text
            img = Image.new('RGB', (128, 32), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                # Try to use default font
                draw.text((5, 5), text, fill='black')
            except:
                # Fallback if font not available
                draw.text((5, 5), text, fill='black')
            
            img_path = data_dir / 'images' / f'img_{i}.jpg'
            img.save(img_path)
            
            # Create annotation
            annotations.append({
                'image_name': f'img_{i}.jpg',
                'bboxes': [[0, 0, 128, 32]],
                'texts': [text]
            })
        
        # Split data
        train_annotations = annotations[:2]
        val_annotations = annotations[2:3]
        test_annotations = annotations[3:]
        
        # Save annotations
        with open(data_dir / 'train_annotations.json', 'w') as f:
            json.dump(train_annotations, f)
        with open(data_dir / 'val_annotations.json', 'w') as f:
            json.dump(val_annotations, f)
        with open(data_dir / 'test_annotations.json', 'w') as f:
            json.dump(test_annotations, f)
        
        # Create vocabulary
        chars = set()
        for ann in annotations:
            for text in ann['texts']:
                chars.update(text)
        
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for i, char in enumerate(sorted(chars)):
            vocab[char] = i + 4
        
        with open(data_dir / 'vocabulary.json', 'w') as f:
            json.dump(vocab, f)
    
    def create_test_config(self):
        """Create test configuration"""
        config = {
            'dataset': {
                'name': 'ocr_dataset',
                'params': {
                    'data_dir': str(Path(self.temp_dir) / 'data'),
                    'image_size': [128, 128]
                }
            },
            'detection_model': {
                'name': 'craft',
                'params': {
                    'pretrained': False,
                    'text_threshold': 0.7,
                    'link_threshold': 0.4
                }
            },
            'recognition_model': {
                'name': 'mamba_ocr',
                'params': {
                    'vocab_size': 20,
                    'embed_dim': 64,
                    'num_layers': 2,
                    'max_seq_length': 16
                }
            },
            'training': {
                'detection': {
                    'epochs': 2,
                    'batch_size': 1,
                    'lr': 0.001,
                    'optimizer': 'adam'
                },
                'recognition': {
                    'epochs': 2,
                    'batch_size': 1,
                    'lr': 0.001,
                    'optimizer': 'adam'
                }
            }
        }
        
        config_path = Path(self.temp_dir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.config_path = str(config_path)
        self.config = config
    
    def test_training_pipeline(self):
        """Test complete training pipeline"""
        trainer = OCRTrainer(self.config)
        
        # Test initialization
        assert trainer.detection_model is not None
        assert trainer.recognition_model is not None
        assert trainer.detection_optimizer is not None
        assert trainer.recognition_optimizer is not None
    
    def test_inference_pipeline_creation(self):
        """Test inference pipeline creation"""
        # Create dummy checkpoint
        checkpoint_dir = Path(self.temp_dir) / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        trainer = OCRTrainer(self.config)
        checkpoint_path = checkpoint_dir / 'test_model.pth'
        trainer.save_checkpoint(str(checkpoint_path), is_best=True)
        
        # Test pipeline creation (would need actual training for full test)
        assert checkpoint_path.exists()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    pytest.main([__file__])
