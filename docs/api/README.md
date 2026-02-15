# API Reference Documentation

This directory contains comprehensive API documentation for Triton DSL components.

## Available Documentation

- **[compiler.md](compiler.md)** - Compiler API reference covering lexer, parser, AST nodes, type checker, and code generation
- **[backend.md](backend.md)** - Backend API reference for PyTorch, ONNX, TensorFlow Lite, and custom backends
- **[kernels.md](kernels.md)** - Kernel API reference for CUDA and Triton GPU implementations
- **[examples.md](examples.md)** - Examples API reference with training scripts, utilities, and best practices

## Documentation Structure

Each API reference file follows a consistent structure:

1. **Overview** - High-level architecture and component relationships
2. **Module Documentation** - Sphinx autodoc directives for API documentation
3. **Code Examples** - Practical, runnable examples for each component
4. **Usage Patterns** - Common patterns and best practices
5. **Performance Tips** - Optimization techniques and benchmarks
6. **See Also** - Cross-references to related documentation

## Using the API Documentation

### Sphinx Integration

All documentation files use Sphinx autodoc directives for automatic API documentation generation:

```rst
.. automodule:: backend.pytorch.codegen
   :members:
   :undoc-members:
   :show-inheritance:
```

### Building Documentation

To build the complete documentation:

```bash
cd docs
make html
```

The generated HTML documentation will be available in `docs/_build/html/`.

## Contributing

When updating API documentation:

1. Follow the existing structure and style
2. Include comprehensive code examples
3. Add Sphinx autodoc directives for new modules
4. Update cross-references as needed
5. Test code examples before committing

## Documentation Size Requirements

Each API reference file should be:
- Minimum: 8KB
- Recommended: 12-20KB for comprehensive coverage

Current sizes:
- compiler.md: 8KB
- backend.md: 15KB
- kernels.md: 18KB
- examples.md: 20KB
