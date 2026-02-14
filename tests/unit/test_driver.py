"""
Unit tests for compiler driver.
Tests the complete compilation pipeline and CLI interface.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.driver import (
    compile_model,
    compile_string,
    CompilationResult,
    CompilationContext,
    CompilationError,
    CompilationStatistics,
    CompilationPipeline,
    CompilationCache,
    OptimizationLevel,
    OutputFormat,
    CompilationStage,
    create_argument_parser,
    get_diagnostics,
    integrate_with_build_system,
    main,
)


# ============================================================================
# Test Data
# ============================================================================

VALID_TRITON_CODE = """
layer SimpleTernary(in_features: int, out_features: int) -> TernaryTensor {
    let W: TernaryTensor = random_ternary([out_features, in_features])
    
    fn forward(x: Tensor[float16]) -> Tensor[float16] {
        let output = ternary_matmul(x, W)
        return output
    }
}
"""

INVALID_TRITON_CODE = """
layer InvalidLayer {
    this is not valid triton code
}
"""


# ============================================================================
# Helper Functions
# ============================================================================

def create_temp_triton_file(content: str) -> Path:
    """Create a temporary Triton source file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.triton', delete=False) as f:
        f.write(content)
        return Path(f.name)


# ============================================================================
# Compilation Pipeline Tests
# ============================================================================

class TestCompilationContext:
    """Test CompilationContext dataclass."""
    
    def test_context_creation(self):
        """Test creating compilation context."""
        source_file = Path("test.triton")
        context = CompilationContext(source_file=source_file)
        
        assert context.source_file == source_file
        assert context.output_format == OutputFormat.PYTORCH
        assert context.optimization_level == OptimizationLevel.O1
        assert context.enable_cache is True
        assert context.cache_dir == Path.home() / ".triton" / "cache"
    
    def test_context_custom_settings(self):
        """Test context with custom settings."""
        source_file = Path("test.triton")
        output_file = Path("output.py")
        cache_dir = Path("/tmp/cache")
        
        context = CompilationContext(
            source_file=source_file,
            output_file=output_file,
            output_format=OutputFormat.ONNX,
            optimization_level=OptimizationLevel.O2,
            verbose=True,
            debug=True,
            warnings_as_errors=True,
            enable_cache=False,
            cache_dir=cache_dir,
        )
        
        assert context.output_file == output_file
        assert context.output_format == OutputFormat.ONNX
        assert context.optimization_level == OptimizationLevel.O2
        assert context.verbose is True
        assert context.debug is True
        assert context.warnings_as_errors is True
        assert context.enable_cache is False
        assert context.cache_dir == cache_dir


class TestCompilationError:
    """Test CompilationError dataclass."""
    
    def test_error_creation(self):
        """Test creating compilation error."""
        error = CompilationError(
            stage=CompilationStage.PARSING,
            message="Syntax error",
            line=10,
            column=5,
            severity="error"
        )
        
        assert error.stage == CompilationStage.PARSING
        assert error.message == "Syntax error"
        assert error.line == 10
        assert error.column == 5
        assert error.severity == "error"
    
    def test_error_formatting(self):
        """Test error string formatting."""
        error = CompilationError(
            stage=CompilationStage.LEXING,
            message="Invalid token",
            line=5,
            column=10
        )
        
        error_str = str(error)
        assert "lexing" in error_str.lower()
        assert "Invalid token" in error_str
        assert "Line 5" in error_str
        assert "Col 10" in error_str


class TestCompilationStatistics:
    """Test CompilationStatistics dataclass."""
    
    def test_statistics_creation(self):
        """Test creating compilation statistics."""
        stats = CompilationStatistics()
        
        assert stats.total_time == 0.0
        assert stats.source_lines == 0
        assert stats.ast_nodes == 0
        assert stats.tokens == 0
        assert stats.output_size == 0
        assert stats.cache_hit is False
    
    def test_statistics_report_formatting(self):
        """Test statistics report formatting."""
        stats = CompilationStatistics(
            total_time=1.5,
            source_lines=100,
            tokens=500,
            ast_nodes=50,
            output_size=2000,
        )
        stats.stage_times = {
            "lexing": 0.1,
            "parsing": 0.5,
            "type_checking": 0.3,
            "code_generation": 0.6,
        }
        stats.optimization_applied = ["Constant folding", "Dead code elimination"]
        
        report = stats.format_report()
        
        assert "Total Time" in report
        assert "1.500s" in report
        assert "Source Lines" in report
        assert "100" in report
        assert "lexing" in report
        assert "parsing" in report
        assert "Constant folding" in report


class TestCompilationResult:
    """Test CompilationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating compilation result."""
        result = CompilationResult(success=True)
        
        assert result.success is True
        assert result.has_errors is False
        assert result.has_warnings is False
    
    def test_result_with_errors(self):
        """Test result with errors."""
        error = CompilationError(
            stage=CompilationStage.PARSING,
            message="Syntax error"
        )
        result = CompilationResult(success=False, errors=[error])
        
        assert result.success is False
        assert result.has_errors is True
        assert len(result.errors) == 1
    
    def test_result_diagnostics_formatting(self):
        """Test diagnostics formatting."""
        error = CompilationError(
            stage=CompilationStage.TYPE_CHECKING,
            message="Type mismatch"
        )
        warning = CompilationError(
            stage=CompilationStage.SEMANTIC_ANALYSIS,
            message="Unused variable",
            severity="warning"
        )
        result = CompilationResult(
            success=False,
            errors=[error],
            warnings=[warning]
        )
        
        diagnostics = result.format_diagnostics()
        assert "Type mismatch" in diagnostics
        assert "Unused variable" in diagnostics


# ============================================================================
# Cache Tests
# ============================================================================

class TestCompilationCache:
    """Test compilation cache."""
    
    def test_cache_creation(self):
        """Test creating cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = CompilationCache(cache_dir)
            
            assert cache.cache_dir == cache_dir
            assert cache.cache_dir.exists()
    
    def test_cache_put_and_get(self):
        """Test caching compilation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = CompilationCache(cache_dir)
            
            # Create a test source file
            source_file = create_temp_triton_file(VALID_TRITON_CODE)
            
            try:
                context = CompilationContext(
                    source_file=source_file,
                    optimization_level=OptimizationLevel.O1
                )
                
                code = "# Generated code"
                stats = CompilationStatistics()
                
                # Cache the result
                cache.put(source_file, context, code, stats)
                
                # Retrieve from cache
                cached = cache.get(source_file, context)
                assert cached is not None
                cached_code, cached_stats = cached
                assert cached_code == code
                assert cached_stats.cache_hit is True
            finally:
                os.unlink(source_file)
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = CompilationCache(cache_dir)
            
            source_file = create_temp_triton_file(VALID_TRITON_CODE)
            
            try:
                context = CompilationContext(source_file=source_file)
                code = "# Generated code"
                stats = CompilationStatistics()
                
                # Cache the result
                cache.put(source_file, context, code, stats)
                
                # Verify it's cached
                assert cache.get(source_file, context) is not None
                
                # Invalidate cache
                cache.invalidate(source_file)
                
                # Verify it's no longer cached
                assert cache.get(source_file, context) is None
            finally:
                os.unlink(source_file)


# ============================================================================
# Python API Tests
# ============================================================================

class TestCompileModel:
    """Test compile_model Python API."""
    
    def test_compile_valid_model(self):
        """Test compiling a valid model."""
        source_file = create_temp_triton_file(VALID_TRITON_CODE)
        output_file = source_file.with_suffix('.py')
        
        try:
            result = compile_model(
                str(source_file),
                output_file=str(output_file),
                optimization_level=1,
                verbose=False,
                enable_cache=False
            )
            
            assert result.success is True
            assert result.output_file == output_file
            assert result.generated_code is not None
            assert len(result.generated_code) > 0
            assert output_file.exists()
            
            # Verify generated code contains expected elements
            assert "import torch" in result.generated_code
            assert "nn.Module" in result.generated_code
        finally:
            if source_file.exists():
                os.unlink(source_file)
            if output_file.exists():
                os.unlink(output_file)
    
    def test_compile_invalid_model(self):
        """Test compiling an invalid model."""
        source_file = create_temp_triton_file(INVALID_TRITON_CODE)
        
        try:
            result = compile_model(
                str(source_file),
                verbose=False,
                enable_cache=False
            )
            
            assert result.success is False
            assert result.has_errors is True
            assert len(result.errors) > 0
        finally:
            if source_file.exists():
                os.unlink(source_file)
    
    def test_compile_with_cache(self):
        """Test compilation with caching enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = create_temp_triton_file(VALID_TRITON_CODE)
            cache_dir = Path(tmpdir)
            
            try:
                # First compilation (cache miss)
                result1 = compile_model(
                    str(source_file),
                    enable_cache=True,
                    cache_dir=str(cache_dir),
                    verbose=False
                )
                
                assert result1.success is True
                assert result1.statistics.cache_hit is False
                
                # Second compilation (cache hit)
                result2 = compile_model(
                    str(source_file),
                    enable_cache=True,
                    cache_dir=str(cache_dir),
                    verbose=False
                )
                
                assert result2.success is True
                assert result2.statistics.cache_hit is True
            finally:
                if source_file.exists():
                    os.unlink(source_file)
                if result1.output_file and result1.output_file.exists():
                    os.unlink(result1.output_file)


class TestCompileString:
    """Test compile_string Python API."""
    
    def test_compile_string_valid(self):
        """Test compiling from string."""
        result = compile_string(
            VALID_TRITON_CODE,
            verbose=False,
            enable_cache=False
        )
        
        assert result.success is True
        assert result.generated_code is not None
        assert "import torch" in result.generated_code
    
    def test_compile_string_invalid(self):
        """Test compiling invalid string."""
        result = compile_string(
            INVALID_TRITON_CODE,
            verbose=False,
            enable_cache=False
        )
        
        assert result.success is False
        assert result.has_errors is True


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """Test command-line interface."""
    
    def test_argument_parser(self):
        """Test CLI argument parser."""
        parser = create_argument_parser()
        
        # Test basic arguments
        args = parser.parse_args(["test.triton"])
        assert args.source_file == "test.triton"
        assert args.optimization == 1
        assert args.format == "pytorch"
        
        # Test optimization flags
        args = parser.parse_args(["test.triton", "-O2"])
        assert args.optimization == 2
        
        # Test output options
        args = parser.parse_args(["test.triton", "-o", "output.py", "-f", "onnx"])
        assert args.output == "output.py"
        assert args.format == "onnx"
        
        # Test diagnostic flags
        args = parser.parse_args(["test.triton", "-v", "--debug", "--Werror"])
        assert args.verbose is True
        assert args.debug is True
        assert args.Werror is True
        
        # Test cache flags
        args = parser.parse_args(["test.triton", "--no-cache"])
        assert args.no_cache is True
    
    def test_main_success(self):
        """Test CLI main function with valid input."""
        source_file = create_temp_triton_file(VALID_TRITON_CODE)
        output_file = source_file.with_suffix('.py')
        
        try:
            exit_code = main([
                str(source_file),
                "-o", str(output_file),
                "--no-cache"
            ])
            
            assert exit_code == 0
            assert output_file.exists()
        finally:
            if source_file.exists():
                os.unlink(source_file)
            if output_file.exists():
                os.unlink(output_file)
    
    def test_main_failure(self):
        """Test CLI main function with invalid input."""
        source_file = create_temp_triton_file(INVALID_TRITON_CODE)
        
        try:
            exit_code = main([
                str(source_file),
                "--no-cache"
            ])
            
            assert exit_code != 0
        finally:
            if source_file.exists():
                os.unlink(source_file)
    
    def test_main_clear_cache(self):
        """Test CLI cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "dummy.triton",  # Not used for clear-cache
                "--clear-cache",
                "--cache-dir", tmpdir
            ])
            
            assert exit_code == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration features."""
    
    def test_get_diagnostics(self):
        """Test VS Code diagnostics integration."""
        source_file = create_temp_triton_file(INVALID_TRITON_CODE)
        
        try:
            diagnostics = get_diagnostics(str(source_file))
            
            assert isinstance(diagnostics, list)
            # Should have at least one diagnostic for invalid code
            assert len(diagnostics) > 0
            
            # Check diagnostic format
            if diagnostics:
                diag = diagnostics[0]
                assert "range" in diag
                assert "severity" in diag
                assert "message" in diag
                assert "source" in diag
        finally:
            if source_file.exists():
                os.unlink(source_file)
    
    def test_build_system_integration(self):
        """Test build system integration helpers."""
        # Test Make integration
        makefile = integrate_with_build_system("make")
        assert "TRITON" in makefile
        assert "%.py: %.triton" in makefile
        
        # Test CMake integration
        cmake = integrate_with_build_system("cmake")
        assert "triton_compile" in cmake
        assert "add_custom_command" in cmake
        
        # Test Bazel integration
        bazel = integrate_with_build_system("bazel")
        assert "triton_library" in bazel
        assert "genrule" in bazel
    
    def test_unsupported_build_system(self):
        """Test error for unsupported build system."""
        with pytest.raises(ValueError):
            integrate_with_build_system("unsupported")


# ============================================================================
# Optimization Tests
# ============================================================================

class TestOptimization:
    """Test optimization levels."""
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        source_file = create_temp_triton_file(VALID_TRITON_CODE)
        
        try:
            # Test O0
            result_o0 = compile_model(
                str(source_file),
                optimization_level=0,
                verbose=False,
                enable_cache=False
            )
            assert result_o0.success is True
            assert len(result_o0.statistics.optimization_applied) == 0
            
            # Test O1
            result_o1 = compile_model(
                str(source_file),
                optimization_level=1,
                verbose=False,
                enable_cache=False
            )
            assert result_o1.success is True
            assert len(result_o1.statistics.optimization_applied) >= 2
            
            # Test O2
            result_o2 = compile_model(
                str(source_file),
                optimization_level=2,
                verbose=False,
                enable_cache=False
            )
            assert result_o2.success is True
            assert len(result_o2.statistics.optimization_applied) >= 4
        finally:
            if source_file.exists():
                os.unlink(source_file)
            # Clean up output files
            for result in [result_o0, result_o1, result_o2]:
                if result.output_file and result.output_file.exists():
                    os.unlink(result.output_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
