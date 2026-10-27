import pytest
import torch
from models.craft_detection import CraftDetection
from models.mamba_recognition import MambaOCR

class TestCraftDetection:
    def test_craft_forward(self):
        model = CraftDetection(pretrained=False)
        x = torch.randn(2, 3, 512, 512)
        
        output, features = model(x)
        
        assert output.shape == (2, 512, 512, 2)
        assert len(features) == 4  # Feature pyramid levels
    
    def test_craft_output_range(self):
        model = CraftDetection(pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        
        output, _ = model(x)
        
        # Output should be in range [0, 1] after sigmoid
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

class TestMambaOCR:
    def test_mamba_forward(self):
        vocab_size = 1000
        model = MambaOCR(vocab_size=vocab_size, embed_dim=256, num_layers=4)
        x = torch.randn(2, 3, 32, 128)
        
        output = model(x)
        
        # Should output sequence of character probabilities
        assert output.shape[0] == 2  # batch size
        assert output.shape[2] == vocab_size  # vocabulary size
    
    def test_mamba_with_different_inputs(self):
        model = MambaOCR(vocab_size=100, embed_dim=128, num_layers=2)
        
        # Test different input sizes
        for h, w in [(32, 64), (32, 128), (32, 256)]:
            x = torch.randn(1, 3, h, w)
            output = model(x)
            assert output.shape[0] == 1
            assert output.shape[2] == 100

if __name__ == '__main__':
    pytest.main([__file__])
