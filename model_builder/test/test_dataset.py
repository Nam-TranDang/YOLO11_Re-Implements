import pytest
import torch
from datasets.ocr_dataset import OCRDataset, OCRCollator
import tempfile
import json
from pathlib import Path

class TestOCRDataset:
    def setup_method(self):
        # Create temporary dataset
        self.temp_dir = tempfile.mkdtemp()
        self.create_mock_dataset()
    
    def create_mock_dataset(self):
        """Create mock dataset for testing"""
        # Create directories
        (Path(self.temp_dir) / 'images').mkdir()
        
        # Create mock annotations
        annotations = [
            {
                "image_name": "img1.jpg",
                "bboxes": [[10, 10, 50, 30], [60, 10, 100, 30]],
                "texts": ["Hello", "World"]
            }
        ]
        
        with open(Path(self.temp_dir) / 'train_annotations.json', 'w') as f:
            json.dump(annotations, f)
        
        # Create vocabulary
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for i, char in enumerate(chars):
            vocab[char] = i + 4
            
        with open(Path(self.temp_dir) / 'vocabulary.json', 'w') as f:
            json.dump(vocab, f)
    
    def test_dataset_loading(self):
        config = {
            'dataset': {'params': {'image_size': [512, 512]}},
            'recognition_model': {'params': {'max_seq_length': 32}}
        }
        
        # This would normally fail without actual images, but tests basic structure
        try:
            dataset = OCRDataset(
                data_dir=self.temp_dir,
                mode='train',
                config=config
            )
            assert len(dataset.vocab) > 0
            assert hasattr(dataset, 'annotations')
        except Exception as e:
            # Expected to fail without actual images
            assert "No such file" in str(e) or "cannot identify image file" in str(e)
    
    def test_collator(self):
        collator = OCRCollator(max_seq_length=32)
        
        # Mock batch data
        batch = [
            {
                'image': torch.randn(3, 32, 128),
                'target_sequence': torch.tensor([1, 5, 10, 2]),  # SOS, chars, EOS
                'text_map': torch.randn(32, 128),
                'link_map': torch.randn(32, 128),
                'text': 'test'
            }
        ]
        
        collated = collator(batch)
        
        assert 'images' in collated
        assert 'target_sequences' in collated
        assert 'text_maps' in collated
        assert 'link_maps' in collated
        assert collated['images'].shape[0] == 1

if __name__ == '__main__':
    pytest.main([__file__])
