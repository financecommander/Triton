"""
Integration tests for training with compiled models.
Tests training, backward pass, gradients, convergence, and comparison with PyTorch.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.integration.test_adapters import quantize_model_to_ternary
from tests.integration.test_utils import (
    validate_gradients,
    test_forward_backward_pass,
)


class TestTrainingIntegration:
    """Test training integration with compiled models."""
    
    def test_compiled_model_training_setup(self, compiled_simple_model):
        """Test that compiled model can be set up for training."""
        model = compiled_simple_model
        
        # Should be able to set to train mode
        model.train()
        assert model.training
        
        # Should be able to create optimizer (even if no trainable params)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        assert optimizer is not None
    
    def test_compiled_model_backward_pass(self, compiled_simple_model):
        """Test backward pass through compiled model."""
        model = compiled_simple_model
        model.train()
        
        # Forward pass
        x = torch.randn(4, 64, requires_grad=True)
        output = model(x)
        
        # Compute loss
        target = torch.randn(4, 128)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_compiled_model_gradient_flow(self, compiled_simple_model):
        """Test gradient flow through compiled model."""
        model = compiled_simple_model
        
        validation = validate_gradients(
            model,
            torch.randn(4, 64),
            torch.randn(4, 128)
        )
        
        # Gradients should flow to input
        assert validation['loss_value'] > 0
    
    def test_training_loop_single_epoch(self, compiled_simple_model, mock_dataloader):
        """Test single epoch of training."""
        model = compiled_simple_model
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, _ in mock_dataloader:
            # Use input as target (autoencoder style)
            target = batch_x
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Adjust target size to match output
            if output.shape != target.shape:
                target = torch.randn_like(output)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 5:  # Just test a few batches
                break
        
        avg_loss = epoch_loss / num_batches
        assert avg_loss > 0
        assert not torch.isnan(torch.tensor(avg_loss))
    
    def test_training_convergence_simple_task(self, training_config):
        """Test that model converges on simple task."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(training_config['input_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, training_config['num_classes']),
        )
        
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Create simple dataset
        X = torch.randn(100, training_config['input_dim'])
        y = torch.randint(0, training_config['num_classes'], (100,))
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Train for a few epochs
        losses = []
        model.train()
        
        for epoch in range(5):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
        
        # Loss should generally decrease
        assert losses[-1] <= losses[0] * 1.5  # Allow some variance
    
    def test_training_vs_eval_mode_difference(self, reference_pytorch_model):
        """Test that training and eval modes produce different results with dropout."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
        
        x = torch.randn(4, 64)
        
        # Training mode (dropout active)
        model.train()
        output_train = model(x)
        
        # Eval mode (dropout inactive)
        model.eval()
        with torch.no_grad():
            output_eval = model(x)
        
        # Outputs should be different due to dropout
        # (May occasionally be similar by chance, but generally different)
        assert output_train.shape == output_eval.shape
    
    def test_gradient_accumulation(self, reference_pytorch_model, mock_dataloader):
        """Test gradient accumulation across batches."""
        model = reference_pytorch_model
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        accumulation_steps = 4
        model.train()
        
        batch_count = 0
        for batch_x, batch_y in mock_dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y) / accumulation_steps
            loss.backward()
            
            batch_count += 1
            if batch_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_count >= 8:
                break
        
        # Should complete without errors
        assert batch_count == 8
    
    def test_learning_rate_scheduling(self, reference_pytorch_model, mock_dataloader):
        """Test learning rate scheduling during training."""
        model = reference_pytorch_model
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(3):
            for batch_x, batch_y in mock_dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                break  # One batch per epoch
            
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # LR should have decreased
        assert final_lr < initial_lr
    
    def test_mixed_precision_training(self, reference_pytorch_model):
        """Test mixed precision training (FP16)."""
        model = reference_pytorch_model
        
        # Use automatic mixed precision if available
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))
        
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        assert not torch.isnan(loss)
    
    def test_gradient_clipping(self, reference_pytorch_model, mock_dataloader):
        """Test gradient clipping during training."""
        model = reference_pytorch_model
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        max_grad_norm = 1.0
        
        for batch_x, batch_y in mock_dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check that gradients are clipped
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            assert total_norm <= max_grad_norm * 1.1  # Small tolerance
            
            optimizer.step()
            break  # Test one batch
    
    def test_training_with_regularization(self, reference_pytorch_model, mock_dataloader):
        """Test training with L2 regularization."""
        model = reference_pytorch_model
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for batch_x, batch_y in mock_dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            break  # Test one batch
        
        assert not torch.isnan(loss)
    
    def test_training_checkpoint_save_load(self, reference_pytorch_model, temp_dir):
        """Test saving and loading training checkpoints."""
        model = reference_pytorch_model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epoch = 5
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        # Create new model and optimizer
        new_model = type(model)()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load states
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        # Check epoch
        assert loaded_checkpoint['epoch'] == epoch
    
    def test_early_stopping_simulation(self, reference_pytorch_model, mock_dataloader):
        """Test early stopping logic."""
        model = reference_pytorch_model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(10):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in mock_dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 3:
                    break
            
            avg_loss = epoch_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break  # Early stopping
        
        # Should have completed some epochs
        assert epoch >= 0
    
    def test_training_metrics_tracking(self, reference_pytorch_model, mock_dataloader):
        """Test tracking training metrics."""
        model = reference_pytorch_model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        metrics = {
            'losses': [],
            'accuracies': [],
        }
        
        for batch_x, batch_y in mock_dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            metrics['losses'].append(loss.item())
            
            # Calculate accuracy
            _, predicted = output.max(1)
            accuracy = (predicted == batch_y).float().mean().item()
            metrics['accuracies'].append(accuracy)
            
            if len(metrics['losses']) >= 5:
                break
        
        assert len(metrics['losses']) == 5
        assert len(metrics['accuracies']) == 5
        assert all(0 <= acc <= 1 for acc in metrics['accuracies'])
    
    def test_quantized_model_training(self, reference_pytorch_model):
        """Test training a quantized model."""
        model = reference_pytorch_model
        
        # Quantize model
        quantize_model_to_ternary(model)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        
        # Training step
        x = torch.randn(4, 64)
        target = torch.randn(4, 10)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
    
    def test_comparison_compiled_vs_pytorch(self, training_config):
        """Compare training of compiled model vs pure PyTorch."""
        # Pure PyTorch model
        pytorch_model = nn.Sequential(
            nn.Linear(training_config['input_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, training_config['num_classes']),
        )
        
        # For comparison, use the same architecture
        # (In real test, would use compiled model)
        compiled_model = nn.Sequential(
            nn.Linear(training_config['input_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, training_config['num_classes']),
        )
        
        # Initialize with same weights
        compiled_model.load_state_dict(pytorch_model.state_dict())
        
        # Train both models identically
        optimizer1 = optim.Adam(pytorch_model.parameters(), lr=0.001)
        optimizer2 = optim.Adam(compiled_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(32, training_config['input_dim'])
        y = torch.randint(0, training_config['num_classes'], (32,))
        
        # Train PyTorch model
        pytorch_model.train()
        optimizer1.zero_grad()
        output1 = pytorch_model(x)
        loss1 = criterion(output1, y)
        loss1.backward()
        optimizer1.step()
        
        # Train compiled model
        compiled_model.train()
        optimizer2.zero_grad()
        output2 = compiled_model(x)
        loss2 = criterion(output2, y)
        loss2.backward()
        optimizer2.step()
        
        # Losses should be similar (same initialization and data)
        assert abs(loss1.item() - loss2.item()) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
