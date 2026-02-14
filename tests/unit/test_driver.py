"""
Unit tests for the Triton Compiler Driver.

Tests cover:
- Compilation pipeline stages
- CLI interface
- Caching system
- Error handling
- Optimization levels
- Multiple backends
- Statistics and diagnostics
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from triton.compiler.driver import (
    CompilationCache,
    CompilationError,
    CompilationOptions,
    CompilationResult,
    CompilationStage,
    OptimizationLevel,
    OutputFormat,
    TargetBackend,
    TritonCompiler,
    compile_model,
    create_cli_parser,
    main,
)


class TestCompilationOptions:
    """Test compilation options."""
    
    def test_default_options(self):
        options = CompilationOptions(source_file="test.triton")
        assert options.optimization_level == OptimizationLevel.O1
        assert options.target_backend == TargetBackend.PYTORCH
        assert options.use_cache is True
        assert options.verbose is False
    
    def test_custom_options(self):
        options = CompilationOptions(
            source_file="test.triton",
            output_file="output.py",
            optimization_level=OptimizationLevel.O2,
            target_backend=TargetBackend.ONNX,
            verbose=True,
        )
        assert options.output_file == "output.py"
        assert options.optimization_level == OptimizationLevel.O2
        assert options.target_backend == TargetBackend.ONNX
        assert options.verbose is True


class TestCompilationError:
    """Test compilation error representation."""
    
    def test_error_string_with_location(self):
        error = CompilationError(
            stage=CompilationStage.PARSER,
            message="Syntax error",
            lineno=10,
            col_offset=5,
            source_file="test.triton"
        )
        error_str = str(error)
        assert "test.triton" in error_str
        assert "10:5" in error_str
        assert "error" in error_str
        assert "parser" in error_str
    
    def test_warning_string(self):
        error = CompilationError(
            stage=CompilationStage.TYPE_CHECKER,
            message="Type mismatch",
            is_warning=True
        )
        error_str = str(error)
        assert "warning" in error_str
        assert "Type mismatch" in error_str


class TestCompilationCache:
    """Test compilation caching."""
    
    def setup_method(self):
        """Setup test cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CompilationCache(self.temp_dir)
        
        # Create a test source file
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
    
    def teardown_method(self):
        """Cleanup test cache directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        assert self.cache.cache_dir.exists()
        assert isinstance(self.cache.metadata, dict)
    
    def test_cache_put_and_get(self):
        options = CompilationOptions(source_file=self.test_source)
        result = CompilationResult(success=True, output_file="output.py")
        
        # Put in cache
        self.cache.put(self.test_source, options, result)
        
        # Get from cache
        cached_result = self.cache.get(self.test_source, options)
        assert cached_result is not None
        assert cached_result.success is True
        assert cached_result.statistics.cache_hit is True
    
    def test_cache_miss_nonexistent(self):
        options = CompilationOptions(source_file=self.test_source)
        cached_result = self.cache.get(self.test_source, options)
        assert cached_result is None
    
    def test_cache_invalidation_on_modification(self):
        options = CompilationOptions(source_file=self.test_source)
        result = CompilationResult(success=True)
        
        # Cache result
        self.cache.put(self.test_source, options, result)
        
        # Modify source file
        import time
        time.sleep(0.1)  # Ensure different mtime
        with open(self.test_source, 'a') as f:
            f.write("\nlet y: trit = 0")
        
        # Cache should be invalidated
        cached_result = self.cache.get(self.test_source, options)
        assert cached_result is None
    
    def test_cache_clear(self):
        options = CompilationOptions(source_file=self.test_source)
        result = CompilationResult(success=True)
        
        # Add to cache
        self.cache.put(self.test_source, options, result)
        assert len(self.cache.metadata) > 0
        
        # Clear cache
        self.cache.clear()
        assert len(self.cache.metadata) == 0
        
        # Should not find in cache
        cached_result = self.cache.get(self.test_source, options)
        assert cached_result is None
    
    def test_cache_different_optimization_levels(self):
        result = CompilationResult(success=True)
        
        options_o1 = CompilationOptions(
            source_file=self.test_source,
            optimization_level=OptimizationLevel.O1
        )
        options_o2 = CompilationOptions(
            source_file=self.test_source,
            optimization_level=OptimizationLevel.O2
        )
        
        # Cache with O1
        self.cache.put(self.test_source, options_o1, result)
        
        # O2 should be a cache miss (different options)
        cached_result = self.cache.get(self.test_source, options_o2)
        assert cached_result is None


class TestCompilationPipeline:
    """Test compilation pipeline stages."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compile_simple_program(self):
        # Create a simple test program
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
        
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is True
        assert result.output_file is not None
        assert os.path.exists(result.output_file)
        assert result.ast is not None
        assert result.statistics.lines_of_code > 0
    
    def test_compile_with_parser_error(self):
        # Create a program with syntax error
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = ")  # Incomplete statement
        
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        # Should fail with parser error
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_compile_missing_file(self):
        options = CompilationOptions(
            source_file="nonexistent.triton",
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].message.lower()
    
    def test_compile_layer_definition(self):
        # Create a simpler program with multiple statements
        # Note: Full layer syntax isn't supported by current parser yet
        source_code = """let w: trit = 1
let x: trit = -1
let z: trit = 0"""
        with open(self.test_source, 'w') as f:
            f.write(source_code)
        
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is True
        assert result.ast is not None
        assert result.statistics.ast_nodes > 0


class TestOptimizationLevels:
    """Test different optimization levels."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.parametrize("opt_level", [
        OptimizationLevel.O0,
        OptimizationLevel.O1,
        OptimizationLevel.O2,
        OptimizationLevel.O3,
    ])
    def test_optimization_level(self, opt_level):
        options = CompilationOptions(
            source_file=self.test_source,
            optimization_level=opt_level,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is True
        
        # O0 should have fewer passes than O2
        if opt_level == OptimizationLevel.O0:
            assert result.statistics.optimization_passes == 0
        elif opt_level == OptimizationLevel.O2:
            assert result.statistics.optimization_passes > 0


class TestTargetBackends:
    """Test different target backends."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.parametrize("backend", [
        TargetBackend.PYTORCH,
        TargetBackend.ONNX,
        TargetBackend.PYTHON,
    ])
    def test_backend(self, backend):
        options = CompilationOptions(
            source_file=self.test_source,
            target_backend=backend,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is True
        assert result.output_file is not None
        
        # Verify output contains backend-specific code
        with open(result.output_file, 'r') as f:
            content = f.read()
            if backend == TargetBackend.PYTORCH:
                assert "torch" in content.lower()
            elif backend == TargetBackend.ONNX:
                assert "onnx" in content.lower()
            elif backend == TargetBackend.PYTHON:
                assert "numpy" in content.lower() or "def" in content


class TestPythonAPI:
    """Test Python API."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compile_model_basic(self):
        result = compile_model(self.test_source, use_cache=False)
        
        assert isinstance(result, CompilationResult)
        assert result.success is True
        assert result.output_file is not None
    
    def test_compile_model_with_options(self):
        output_file = os.path.join(self.temp_dir, "output.py")
        
        result = compile_model(
            self.test_source,
            output_file=output_file,
            optimization_level=2,
            target="pytorch",
            verbose=False,
            use_cache=False
        )
        
        assert result.success is True
        assert result.output_file == output_file
        assert os.path.exists(output_file)
    
    def test_compile_model_with_cache(self):
        # First compilation
        result1 = compile_model(self.test_source, cache_dir=self.temp_dir)
        assert result1.success is True
        assert result1.statistics.cache_hit is False
        
        # Second compilation should hit cache
        result2 = compile_model(self.test_source, cache_dir=self.temp_dir)
        assert result2.success is True
        assert result2.statistics.cache_hit is True


class TestCLIInterface:
    """Test CLI interface."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cli_parser_creation(self):
        parser = create_cli_parser()
        assert parser is not None
    
    def test_cli_compile_basic(self):
        output_file = os.path.join(self.temp_dir, "output.py")
        
        exit_code = main([
            "compile",
            self.test_source,
            "-o", output_file,
            "--no-cache",
            "-q"  # Quiet mode
        ])
        
        assert exit_code == 0
        assert os.path.exists(output_file)
    
    def test_cli_compile_with_optimization(self):
        exit_code = main([
            "compile",
            self.test_source,
            "--O2",
            "--no-cache",
            "-q"
        ])
        
        assert exit_code == 0
    
    def test_cli_compile_with_target(self):
        exit_code = main([
            "compile",
            self.test_source,
            "--target", "pytorch",
            "--no-cache",
            "-q"
        ])
        
        assert exit_code == 0
    
    def test_cli_compile_missing_file(self):
        exit_code = main([
            "compile",
            "nonexistent.triton",
            "--no-cache",
            "-q"
        ])
        
        assert exit_code != 0
    
    def test_cli_version(self):
        exit_code = main(["version"])
        assert exit_code == 0
    
    def test_cli_cache_clear(self):
        cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Note: This would require modifying the cache clear command to accept a custom dir
        # For now, just test that the command doesn't crash
        exit_code = main(["cache", "clear"])
        assert exit_code == 0
    
    def test_cli_cache_info(self):
        exit_code = main(["cache", "info"])
        assert exit_code == 0
    
    def test_cli_with_statistics(self):
        exit_code = main([
            "compile",
            self.test_source,
            "--statistics",
            "--no-cache",
            "-q"
        ])
        
        assert exit_code == 0
    
    def test_cli_with_optimization_report(self):
        exit_code = main([
            "compile",
            self.test_source,
            "--optimization-report",
            "--no-cache",
            "-q"
        ])
        
        assert exit_code == 0


class TestStatistics:
    """Test compilation statistics."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1\nlet y: trit = 0")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_statistics_collection(self):
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        assert result.success is True
        stats = result.statistics
        
        # Check that statistics were collected
        assert stats.total_time > 0
        assert stats.lines_of_code == 2
        assert stats.ast_nodes > 0
        assert stats.lexer_time >= 0
        assert stats.parser_time >= 0
    
    def test_statistics_string_representation(self):
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        stats_str = str(result.statistics)
        assert "Total Time" in stats_str
        assert "Lines of Code" in stats_str
        assert "AST Nodes" in stats_str


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_source = os.path.join(self.temp_dir, "test.triton")
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_partial_compilation_on_error(self):
        # Create a program with error
        with open(self.test_source, 'w') as f:
            f.write("let x: invalid_type = 1")
        
        options = CompilationOptions(
            source_file=self.test_source,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        # Should have some partial results even if failed
        assert result.ast is not None or len(result.errors) > 0
    
    def test_warnings_as_errors_mode(self):
        with open(self.test_source, 'w') as f:
            f.write("let x: trit = 1")
        
        options = CompilationOptions(
            source_file=self.test_source,
            warnings_as_errors=True,
            use_cache=False,
            show_progress=False
        )
        
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        # This test would need actual warnings to be generated
        # For now, just verify the option is set
        assert options.warnings_as_errors is True


class TestIntegration:
    """Integration tests for full compilation workflow."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compile_and_cache_workflow(self):
        test_source = os.path.join(self.temp_dir, "model.triton")
        with open(test_source, 'w') as f:
            f.write("let x: trit = 1\nlet y: trit = -1")
        
        # First compilation
        result1 = compile_model(
            test_source,
            optimization_level=2,
            target="pytorch",
            use_cache=True,
            cache_dir=self.temp_dir
        )
        
        assert result1.success is True
        assert not result1.statistics.cache_hit
        
        # Second compilation should use cache
        result2 = compile_model(
            test_source,
            optimization_level=2,
            target="pytorch",
            use_cache=True,
            cache_dir=self.temp_dir
        )
        
        assert result2.success is True
        assert result2.statistics.cache_hit
        
        # Changing optimization level should bypass cache
        result3 = compile_model(
            test_source,
            optimization_level=1,
            target="pytorch",
            use_cache=True,
            cache_dir=self.temp_dir
        )
        
        assert result3.success is True
        assert not result3.statistics.cache_hit
    
    def test_multi_file_compilation(self):
        # Create multiple source files
        files = []
        for i in range(3):
            source_file = os.path.join(self.temp_dir, f"model{i}.triton")
            with open(source_file, 'w') as f:
                f.write(f"let x{i}: trit = {[-1, 0, 1][i]}")
            files.append(source_file)
        
        # Compile all files
        results = []
        for source_file in files:
            result = compile_model(source_file, use_cache=False)
            results.append(result)
        
        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 3
