"""
Unit tests for enhanced CIFAR-10 training script

Tests the new features added to train_ternary_models.py:
- CutMix augmentation
- MixUp augmentation  
- Label smoothing
- Early stopping
- Checkpoint saving/loading with scheduler state
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path

# Mock torch before importing the training script
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()
sys.modules['torch.multiprocessing'] = MagicMock()
sys.modules['torch.nn.parallel'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.datasets'] = MagicMock()

# Mock model imports
sys.modules['models'] = MagicMock()
sys.modules['models.resnet18'] = MagicMock()
sys.modules['models.resnet18.ternary_resnet18'] = MagicMock()
sys.modules['models.mobilenetv2'] = MagicMock()
sys.modules['models.mobilenetv2.ternary_mobilenetv2'] = MagicMock()


class TestCutMix(unittest.TestCase):
    """Test CutMix augmentation"""
    
    def test_cutmix_initialization(self):
        """Test CutMix can be initialized"""
        # Import after mocking
        from models.scripts.train_ternary_models import CutMix
        
        cutmix = CutMix(alpha=1.0)
        self.assertEqual(cutmix.alpha, 1.0)
    
    def test_cutmix_bbox_generation(self):
        """Test bounding box generation"""
        from models.scripts.train_ternary_models import CutMix
        
        cutmix = CutMix(alpha=1.0)
        
        # Test with 32x32 images
        size = (16, 3, 32, 32)
        lam = 0.5
        
        bbx1, bby1, bbx2, bby2 = cutmix._rand_bbox(size, lam)
        
        # Check bounds
        self.assertGreaterEqual(bbx1, 0)
        self.assertGreaterEqual(bby1, 0)
        self.assertLessEqual(bbx2, 32)
        self.assertLessEqual(bby2, 32)
        self.assertLess(bbx1, bbx2)
        self.assertLess(bby1, bby2)


class TestMixUp(unittest.TestCase):
    """Test MixUp augmentation"""
    
    def test_mixup_initialization(self):
        """Test MixUp can be initialized"""
        from models.scripts.train_ternary_models import MixUp
        
        mixup = MixUp(alpha=1.0)
        self.assertEqual(mixup.alpha, 1.0)


class TestLabelSmoothing(unittest.TestCase):
    """Test label smoothing loss"""
    
    def test_label_smoothing_initialization(self):
        """Test label smoothing can be initialized"""
        from models.scripts.train_ternary_models import LabelSmoothingCrossEntropy
        
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.assertEqual(criterion.smoothing, 0.1)
    
    def test_label_smoothing_range(self):
        """Test label smoothing values are in valid range"""
        from models.scripts.train_ternary_models import LabelSmoothingCrossEntropy
        
        # Valid ranges
        for smoothing in [0.0, 0.05, 0.1, 0.15, 0.2]:
            criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
            self.assertGreaterEqual(criterion.smoothing, 0.0)
            self.assertLessEqual(criterion.smoothing, 1.0)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping functionality"""
    
    def test_early_stopping_initialization(self):
        """Test early stopping can be initialized"""
        from models.scripts.train_ternary_models import EarlyStopping
        
        es = EarlyStopping(patience=5, min_delta=0.01, mode='max')
        self.assertEqual(es.patience, 5)
        self.assertEqual(es.min_delta, 0.01)
        self.assertEqual(es.mode, 'max')
        self.assertEqual(es.counter, 0)
        self.assertIsNone(es.best_score)
        self.assertFalse(es.should_stop)
    
    def test_early_stopping_improves(self):
        """Test early stopping when score improves"""
        from models.scripts.train_ternary_models import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='max')
        
        # First score
        should_stop = es(85.0)
        self.assertFalse(should_stop)
        self.assertEqual(es.best_score, 85.0)
        self.assertEqual(es.counter, 0)
        
        # Improvement
        should_stop = es(86.0)
        self.assertFalse(should_stop)
        self.assertEqual(es.best_score, 86.0)
        self.assertEqual(es.counter, 0)
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping when score doesn't improve"""
        from models.scripts.train_ternary_models import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='max')
        
        # Initial scores
        es(85.0)
        es(85.0)  # No improvement, counter=1
        es(84.5)  # Worse, counter=2
        should_stop = es(85.0)  # No improvement, counter=3
        
        self.assertTrue(es.should_stop)
        self.assertTrue(should_stop)
        self.assertEqual(es.counter, 3)
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (for loss)"""
        from models.scripts.train_ternary_models import EarlyStopping
        
        es = EarlyStopping(patience=2, mode='min')
        
        # Initial
        es(1.0)
        self.assertEqual(es.best_score, 1.0)
        
        # Improvement (lower is better)
        es(0.8)
        self.assertEqual(es.best_score, 0.8)
        self.assertEqual(es.counter, 0)
        
        # No improvement
        es(0.9)
        self.assertEqual(es.counter, 1)


class TestGetDataset(unittest.TestCase):
    """Test dataset loading with augmentation options"""
    
    @patch('models.scripts.train_ternary_models.torchvision.datasets.CIFAR10')
    @patch('models.scripts.train_ternary_models.transforms')
    def test_cifar10_train(self, mock_transforms, mock_cifar10):
        """Test CIFAR-10 training dataset loading"""
        from models.scripts.train_ternary_models import get_dataset
        
        # Mock transforms
        mock_transforms.Compose = Mock(return_value='transform')
        mock_transforms.RandomCrop = Mock()
        mock_transforms.RandomHorizontalFlip = Mock()
        mock_transforms.ToTensor = Mock()
        mock_transforms.Normalize = Mock()
        
        dataset, num_classes = get_dataset('cifar10', train=True)
        
        self.assertEqual(num_classes, 10)
        mock_cifar10.assert_called_once()
    
    @patch('models.scripts.train_ternary_models.torchvision.datasets.CIFAR10')
    @patch('models.scripts.train_ternary_models.transforms')
    def test_cifar10_val(self, mock_transforms, mock_cifar10):
        """Test CIFAR-10 validation dataset loading"""
        from models.scripts.train_ternary_models import get_dataset
        
        mock_transforms.Compose = Mock(return_value='transform')
        mock_transforms.ToTensor = Mock()
        mock_transforms.Normalize = Mock()
        
        dataset, num_classes = get_dataset('cifar10', train=False)
        
        self.assertEqual(num_classes, 10)
        mock_cifar10.assert_called_once()


class TestSaveCheckpoint(unittest.TestCase):
    """Test checkpoint saving with enhanced features"""
    
    def test_checkpoint_structure(self):
        """Test checkpoint contains all required fields"""
        from models.scripts.train_ternary_models import save_checkpoint
        
        # Create mock objects
        model = Mock()
        model.state_dict = Mock(return_value={'weight': 'data'})
        
        optimizer = Mock()
        optimizer.state_dict = Mock(return_value={'lr': 0.1})
        
        scheduler = Mock()
        scheduler.state_dict = Mock(return_value={'step': 10})
        
        early_stopping = Mock()
        early_stopping.counter = 5
        early_stopping.best_score = 85.0
        early_stopping.patience = 40
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock torch.save
            with patch('models.scripts.train_ternary_models.torch.save') as mock_save:
                save_checkpoint(model, optimizer, scheduler, 10, 0.5, 85.0, 
                              temp_path, early_stopping)
                
                # Check save was called
                mock_save.assert_called_once()
                
                # Check checkpoint structure
                checkpoint = mock_save.call_args[0][0]
                self.assertIn('epoch', checkpoint)
                self.assertIn('model_state_dict', checkpoint)
                self.assertIn('optimizer_state_dict', checkpoint)
                self.assertIn('scheduler_state_dict', checkpoint)
                self.assertIn('early_stopping', checkpoint)
                
                # Check values
                self.assertEqual(checkpoint['epoch'], 10)
                self.assertEqual(checkpoint['accuracy'], 85.0)
                self.assertEqual(checkpoint['early_stopping']['counter'], 5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestTrainingScript(unittest.TestCase):
    """Test overall training script structure"""
    
    def test_imports(self):
        """Test all imports are available"""
        try:
            from models.scripts import train_ternary_models
            self.assertTrue(hasattr(train_ternary_models, 'CutMix'))
            self.assertTrue(hasattr(train_ternary_models, 'MixUp'))
            self.assertTrue(hasattr(train_ternary_models, 'LabelSmoothingCrossEntropy'))
            self.assertTrue(hasattr(train_ternary_models, 'EarlyStopping'))
            self.assertTrue(hasattr(train_ternary_models, 'get_dataset'))
            self.assertTrue(hasattr(train_ternary_models, 'train_epoch'))
            self.assertTrue(hasattr(train_ternary_models, 'validate'))
            self.assertTrue(hasattr(train_ternary_models, 'save_checkpoint'))
            self.assertTrue(hasattr(train_ternary_models, 'main'))
        except ImportError as e:
            self.fail(f"Import failed: {e}")


if __name__ == '__main__':
    unittest.main()
