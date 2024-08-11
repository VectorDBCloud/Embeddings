![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-green.svg)
![License](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)

# Open-Source Embeddings

Welcome to the Vector Database Cloud Open-Source Embeddings repository! This repository provides pre-computed embeddings for various datasets and models, optimized for use with vector databases. The embeddings cover a range of data types, including text, images, and more, facilitating applications like semantic search, classification, and clustering.

## Table of Contents

1. [About Vector Database Cloud](#about-vector-database-cloud)
2. [Introduction](#introduction)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
5. [Embeddings](#embeddings)
   1. [Text Embeddings](#text-embeddings)
   2. [Image Embeddings](#image-embeddings)
   3. [Multimodal Embeddings](#multimodal-embeddings)
6. [Installation](#installation)
7. [Troubleshooting](#troubleshooting)
8. [Code of Conduct](#code-of-conduct)
9. [Related Resources](#related-resources)
10. [Contributing](#contributing)
11. [License](#license)
12. [Disclaimer](#disclaimer)


## About Vector Database Cloud

[Vector Database Cloud](https://vectordbcloud.com) is a platform that provides one-click deployment of popular vector databases including Qdrant, Milvus, ChromaDB, and Pgvector on cloud. Our platform ensures a secure API, a comprehensive customer dashboard, efficient vector search, and real-time monitoring.

## Introduction

Vector Database Cloud is designed to seamlessly integrate with your existing data workflows. Whether you're working with structured data, unstructured data, or high-dimensional vectors, you can leverage popular ETL (Extract, Transform, Load) tools to streamline the process of moving data into and out of Vector Database Cloud.

This repository serves as a centralized resource for pre-computed embeddings that can be directly used with vector databases. These embeddings are generated using popular models and datasets, and are intended to save time and computational resources for users.

## Prerequisites

- Python 3.7+
- Sufficient storage space for downloading embeddings

## Usage

To use these pre-computed embeddings:

1. Click on the "Download Embeddings" link for the desired embedding set.
2. Save the downloaded file to your local machine or server.
3. Use a compatible library (e.g., numpy for numerical arrays, or specific libraries mentioned in the usage instructions) to load the embeddings into your project.


## Embeddings

To use these pre-computed embeddings:

1. Click on the "Download Embeddings" link for the desired embedding set.
2. Save the downloaded file to your local machine or server.
3. Use a compatible library (e.g., numpy for numerical arrays, or specific libraries mentioned in the usage instructions) to load the embeddings into your project.

### Text Embeddings

- **[BERT Base Uncased](https://huggingface.co/bert-base-uncased)**
    *Description*: Pre-computed embeddings from the BERT Base Uncased model, ideal for NLP tasks requiring context-aware representations.
    *Usage*: Suitable for semantic search and sentence classification.
    *Link to Embeddings*: [Download Embeddings](https://example.com/bert-base-uncased-embeddings)
    *Loading Instructions*: 
    ```python
    import numpy as np
    embeddings = np.load('path/to/bert-base-uncased-embeddings.npy')
    ```

- **[Sentence-BERT (SBERT)](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)**
    *Description*: Sentence embeddings derived from SBERT, optimized for semantic textual similarity tasks.
    *Usage*: Effective for sentence-level semantic search and clustering.
    *Link to Embeddings*: [Download Embeddings](https://example.com/sbert-embeddings)
    *Loading Instructions*: 
    ```python
    import numpy as np
    embeddings = np.load('path/to/sbert-embeddings.npy')
    ```

### Image Embeddings

- **[ResNet-50](https://huggingface.co/microsoft/resnet-50)**
    *Description*: Embeddings generated from the ResNet-50 model, commonly used for image recognition tasks.
    *Usage*: Ideal for image similarity search and object recognition.
    *Link to Embeddings*: [Download Embeddings](https://example.com/resnet-50-embeddings)
    *Loading Instructions*: 
    ```python
    import torch
    embeddings = torch.load('path/to/resnet-50-embeddings.pt')
    ```

- **[CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)**
    *Description*: Multimodal embeddings from CLIP, which encodes images and text into a shared embedding space.
    *Usage*: Useful for cross-modal retrieval tasks.
    *Link to Embeddings*: [Download Embeddings](https://example.com/clip-vit-b32-embeddings)
    *Loading Instructions*: 
    ```python
    import torch
    embeddings = torch.load('path/to/clip-vit-b32-embeddings.pt')
    ```

### Multimodal Embeddings

- **[VisualBERT](https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre)**
    *Description*: Embeddings that combine visual and textual information, suitable for multimodal tasks.
    *Usage*: Good for visual question answering and captioning.
    *Link to Embeddings*: [Download Embeddings](https://example.com/visualbert-embeddings)
    *Loading Instructions*: 
    ```python
    import torch
    embeddings = torch.load('path/to/visualbert-embeddings.pt')
    ```

- **[VilBERT](https://huggingface.co/facebook/vilbert-multi-task)**
    *Description*: Embeddings from the VilBERT model, designed for understanding visual and language inputs together.
    *Usage*: Effective for tasks that require joint vision-language understanding.
    *Link to Embeddings*: [Download Embeddings](https://example.com/vilbert-embeddings)
    *Loading Instructions*: 
    ```python
    import torch
    embeddings = torch.load('path/to/vilbert-embeddings.pt')
    ```
```

## Installation

To use these pre-computed embeddings in your project, follow these steps:

1. Clone this repository:
   ```
   git clone https://github.com/VectorDBCloud/Embeddings.git
   cd Embeddings
   ```

2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the desired embedding files using the links provided in the [Embeddings](#embeddings) section.

5. Place the downloaded embedding files in the appropriate directory within the project structure.

## Common Issues

1. **File Not Found Error**: Ensure that you've downloaded the embedding files and placed them in the correct directory.

2. **Memory Error**: Some embedding files are large and may cause memory issues on machines with limited RAM. Consider using a machine with more memory or processing the embeddings in smaller batches.

3. **Incompatible Numpy Version**: If you encounter issues loading the .npy files, ensure that you're using a compatible version of numpy. You can update numpy using `pip install --upgrade numpy`.

4. **Torch Version Mismatch**: For embeddings stored in .pt format, make sure your PyTorch version is compatible with the saved tensors. You may need to update PyTorch using `pip install --upgrade torch`.

5. **Slow Loading Times**: Large embedding files may take a while to load. This is normal, especially for the first load. Subsequent loads may be faster if the file is cached by your operating system.

## Troubleshooting

If you encounter issues while using these embeddings, try the following steps:

1. **Check Your Python Version**: Ensure you're using Python 3.7 or higher. You can check your version with `python --version`.

2. **Verify File Integrity**: If you're having issues loading an embedding file, try re-downloading it. The file may have been corrupted during the download process.

3. **Check Available Disk Space**: Ensure you have enough free disk space to store and work with the embedding files.

4. **Update Dependencies**: Make sure all your dependencies are up to date. You can update all packages listed in requirements.txt with:
   ```
   pip install --upgrade -r requirements.txt
   ```

5. **Loading Large Files**: If you're working with particularly large embedding files, consider using memory-mapping or lazy loading techniques. For example, with numpy:
   ```python
   import numpy as np
   embeddings = np.load('path/to/large_embeddings.npy', mmap_mode='r')
   ```
6. **GPU vs CPU**: If you're working with PyTorch tensors and have a GPU available, you might want to move the embeddings to the GPU for faster processing:
   ```python
   import torch
   embeddings = torch.load('path/to/embeddings.pt')
   if torch.cuda.is_available():
       embeddings = embeddings.cuda()
   ```
   Conversely, if you're getting CUDA out of memory errors, you may need to move the embeddings to CPU:
   ```python
   embeddings = embeddings.cpu()
   ```

7. **Embedding Dimension Mismatch**: If you're getting dimension mismatch errors when using the embeddings, double-check that you're using the correct embedding file for your task and that your model is configured to expect the correct embedding size.

8. **File Permissions**: Ensure you have the necessary read permissions for the embedding files. On Unix-based systems, you can change permissions with:
   ```
   chmod 644 path/to/embedding_file
   ```

9. **Encoding Issues**: If you encounter encoding errors when loading text-based embedding files, try specifying the encoding explicitly:
   ```python
   with open('path/to/embeddings.txt', 'r', encoding='utf-8') as f:
       # Your loading code here
   ```

10. **Out of Memory Errors**: For very large embedding sets, you might need to process them in batches. Consider implementing a generator or iterator to load and process embeddings in smaller chunks.

If you've tried these troubleshooting steps and are still experiencing issues, please open an issue in the repository with a detailed description of the problem, including any error messages and the steps to reproduce the issue.



## Code of Conduct

We adhere to the [Vector Database Cloud Code of Conduct](https://github.com/VectorDBCloud/Community/blob/main/CODE_OF_CONDUCT.md). Please ensure contributions align with our community standards.


## Related Resources  

- [Vector Database Cloud Documentation](https://docs.vectordbcloud.com)
- [Embeddings Repository](https://github.com/VectorDBCloud/Embeddings)
- [Vector Database Benchmarks](https://github.com/VectorDBCloud/Benchmarks)
- [Vector Database Use Cases](https://github.com/VectorDBCloud/Use-Cases)
- [Community Forum](https://community.vectordbcloud.com)
-

## Contributing

We welcome contributions to improve and expand our Open-Source Embedding Cookbook! Here's how you can contribute:  
1. **Fork the repository**: Create your own fork of the code.
2. **Create a new branch**: Make your changes in a new git branch.
3. **Make your changes**: Enhance existing cookbooks or add new ones.
4. **Follow the style guidelines**: Ensure your code follows our coding standards.
5. **Write clear commit messages**: Your commit messages should clearly describe the changes you've made.
6. **Submit a pull request**: Open a new pull request with your changes.
7. **Respond to feedback**: Be open to feedback and make necessary adjustments to your pull request.

For more detailed information on contributing, please refer to our [Contribution Guidelines](CONTRIBUTING.md).  


## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).

Copyright (c) 2024 Vector Database Cloud

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution — You must give appropriate credit to Vector Database Cloud, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests Vector Database Cloud endorses you or your use.

Additionally, we require that any use of this guide includes visible attribution to Vector Database Cloud. This attribution should be in the form of "Embeddings curated by Vector Database Cloud" or "Based on Vector Database Cloud Embeddings", along with a link to https://vectordbcloud.com, in any public-facing applications, documentation, or redistributions of this guide.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For the full license text, visit: https://creativecommons.org/licenses/by/4.0/legalcode


## Disclaimer

The information and resources provided in this community repository are for general informational purposes only. While we strive to keep the information up-to-date and correct, we make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the information, products, services, or related graphics contained in this repository for any purpose. Any reliance you place on such information is therefore strictly at your own risk.

Vector Database Cloud configurations may vary, and it's essential to consult the official documentation before implementing any solutions or suggestions found in this community repository. Always follow best practices for security and performance when working with databases and cloud services.

The content in this repository may change without notice. Users are responsible for ensuring they are using the most current version of any information or code provided.

This disclaimer applies to Vector Database Cloud, its contributors, and any third parties involved in creating, producing, or delivering the content in this repository.

The use of any information or code in this repository may carry inherent risks, including but not limited to data loss, system failures, or security vulnerabilities. Users should thoroughly test and validate any implementations in a safe environment before deploying to production systems.

For complex implementations or critical systems, we strongly recommend seeking advice from qualified professionals or consulting services.

By using this repository, you acknowledge and agree to this disclaimer. If you do not agree with any part of this disclaimer, please do not use the information or resources provided in this repository.
