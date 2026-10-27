# Not Touch Yet - Unused file - Check Schema  27/Oct/2026
import json
import jsonschema
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Any

class DataValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.annotation_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_name": {"type": "string"},
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        }
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["image_name", "bboxes", "texts"]
            }
        }
        
    def validate_annotations(self, annotation_file: str) -> bool:
        """Validate annotation file format"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            jsonschema.validate(data, self.annotation_schema)
            
            # Additional validation
            for item in data:
                if len(item['bboxes']) != len(item['texts']):
                    raise ValueError(f"Bbox and text count mismatch in {item['image_name']}")
                    
                # Validate bbox coordinates
                for bbox in item['bboxes']:
                    x1, y1, x2, y2 = bbox
                    if x1 >= x2 or y1 >= y2:
                        raise ValueError(f"Invalid bbox coordinates in {item['image_name']}")
                        
            return True
            
        except Exception as e:
            print(f"Annotation validation failed: {e}")
            return False
            
    def validate_images(self, data_dir: str, annotations: List[Dict]) -> bool:
        """Validate image files"""
        image_dir = Path(data_dir) / 'images'
        missing_images = []
        corrupt_images = []
        
        for item in annotations:
            image_path = image_dir / item['image_name']
            
            # Check if image exists
            if not image_path.exists():
                missing_images.append(item['image_name'])
                continue
                
            # Check if image can be loaded
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    corrupt_images.append(item['image_name'])
                    continue
                    
                # Check image dimensions
                h, w = image.shape[:2]
                if h < 32 or w < 32:
                    print(f"Warning: Image {item['image_name']} is too small ({w}x{h})")
                    
            except Exception as e:
                corrupt_images.append(item['image_name'])
                print(f"Error loading {item['image_name']}: {e}")
                
        if missing_images:
            print(f"Missing images: {missing_images}")
            
        if corrupt_images:
            print(f"Corrupt images: {corrupt_images}")
            
        return len(missing_images) == 0 and len(corrupt_images) == 0
        
    def validate_vocabulary(self, vocab_file: str) -> bool:
        """Validate vocabulary file"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
                
            required_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
            for token in required_tokens:
                if token not in vocab:
                    print(f"Missing required token: {token}")
                    return False
                    
            # Check for duplicate indices
            indices = list(vocab.values())
            if len(indices) != len(set(indices)):
                print("Duplicate indices in vocabulary")
                return False
                
            return True
            
        except Exception as e:
            print(f"Vocabulary validation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Validate OCR dataset')
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--config', required=True, help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    validator = DataValidator(config)
    
    # Validate annotations
    splits = ['train', 'val', 'test']
    all_valid = True
    
    for split in splits:
        annotation_file = Path(args.data_dir) / f'{split}_annotations.json'
        if annotation_file.exists():
            print(f"Validating {split} annotations...")
            if not validator.validate_annotations(str(annotation_file)):
                all_valid = False
            else:
                print(f"✅ {split} annotations are valid")
                
                # Validate corresponding images
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                    
                if validator.validate_images(args.data_dir, annotations):
                    print(f"✅ {split} images are valid")
                else:
                    print(f"❌ {split} images validation failed")
                    all_valid = False
        else:
            print(f"⚠️  {split} annotations not found")
    
    # Validate vocabulary
    vocab_file = Path(args.data_dir) / 'vocabulary.json'
    if vocab_file.exists():
        if validator.validate_vocabulary(str(vocab_file)):
            print("✅ Vocabulary is valid")
        else:
            print("❌ Vocabulary validation failed")
            all_valid = False
    
    if all_valid:
        print("\n🎉 All validations passed!")
    else:
        print("\n❌ Some validations failed!")
        exit(1)

if __name__ == '__main__':
    main()
