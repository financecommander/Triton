.. Triton DSL documentation master file

Welcome to Triton DSL's Documentation!
=======================================

**Triton DSL** is a high-performance Domain-Specific Language designed to optimize Ternary Neural Networks by enforcing ternary constraints ({-1, 0, 1}) at the syntax level. This enables 20-40% memory density improvements over standard FP32 representations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   ../getting_started

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   ../tutorials/01_basic_model
   ../tutorials/02_quantization
   ../tutorials/03_custom_layers
   ../tutorials/04_training
   ../tutorials/05_deployment
   ../tutorials/06_advanced_features

.. toctree::
   :maxdepth: 2
   :caption: DSL Reference

   ../dsl/language_spec
   ../dsl/syntax_guide
   ../dsl/type_system
   ../dsl/builtin_functions
   ../dsl/quantization_primitives

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   ../api/compiler
   ../api/backend
   ../api/kernels
   ../api/examples

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   ../architecture/compiler_pipeline
   ../architecture/type_checker
   ../architecture/code_generator
   ../architecture/optimization_passes
   ../architecture/extension_points

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   ../benchmarks/performance
   ../benchmarks/model_zoo
   ../benchmarks/quantization_accuracy
   ../benchmarks/compilation_speed

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   ../contributing/development_setup
   ../contributing/code_style
   ../contributing/testing
   ../contributing/pr_process

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
