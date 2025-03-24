# 🎙️ CSM-1B Gradio Interface 🎧

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/sesame/csm-1b)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A user-friendly Gradio interface for [Sesame's CSM-1B model](https://huggingface.co/sesame/csm-1b) that allows you to easily generate conversations and monologues using Conversational Speech Model technology.

## ✨ Features

- 🗣️ **Multi-Speaker Conversations**: Generate natural-sounding conversations between up to 10 speakers
- 🎭 **Voice Cloning**: Upload your own voice samples to personalize the generated speech
- 🔊 **Built-in Voices**: Use the included reference voices or generate random voices
- 📝 **Simple JSON Input**: Easily format your conversations with a simple JSON structure
- 🎚️ **Advanced Controls**: Fine-tune generation parameters like temperature and audio length
- 🌐 **Web Interface**: Intuitive UI powered by Gradio

## 🚀 Quick Start

### Prerequisites

- Python 3.10 (recommended)
- CUDA-compatible GPU (for optimal performance)
- `ffmpeg` installed on your system

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/c3-csm-gradio.git
   cd c3-csm-gradio
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face (to access the model):
   ```bash
   huggingface-cli login
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:7860`

## 🧩 Usage Examples

### Conversation Mode

Create conversations between multiple speakers using this JSON format:

```json
[
  {"speaker_id": 0, "text": "This voice synthesis is amazing!"},
  {"speaker_id": 1, "text": "I agree, it sounds so natural!"},
  {"speaker_id": 2, "text": "And it's simple to customize voices too."}
]
```

### Monologue Mode

Generate a speech from a single speaker:

```json
[
  "Welcome to my presentation.",
  "Today we'll explore the future of AI speech synthesis.",
  "Let's begin with the fundamentals."
]
```

## 🐳 Docker Support

### Using Pre-built Image

Pull and run the pre-built Docker image from GitHub Container Registry:

```bash
# Pull the image
docker pull ghcr.io/comput3ai/c3-csm-gradio:latest

# Run the container with your Hugging Face token
docker run -p 7860:7860 --gpus all -e HF_TOKEN=your_huggingface_token ghcr.io/comput3ai/c3-csm-gradio
```

### Building Locally

Build and run the application using Docker:

```bash
# Build the image
docker build -t csm-gradio .

# Run the container with your Hugging Face token
docker run -p 7860:7860 --gpus all -e HF_TOKEN=your_huggingface_token csm-gradio
```

### About HF_TOKEN

The `HF_TOKEN` environment variable is required for the container to authenticate with Hugging Face Hub and download the model files. You can obtain this token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

## ⚙️ Advanced Configuration

- **Temperature**: Controls randomness (0.1-2.0, default: 0.9)
- **Top-k**: Limits token selection (1-100, default: 50)
- **Max Audio Length**: Maximum duration per utterance (1000-30000ms)
- **Pause Duration**: Silence between utterances (0-1000ms)

## 🔍 Implementation Details

This application is built on:

- **CSM-1B Model**: Sesame's Conversational Speech Model
- **Llama-3.2-1B**: For text processing
- **Mimi**: For audio codec operations
- **Gradio**: For the web interface

## ⚠️ Ethical Use Guidelines

This tool is provided for research, education, and legitimate creative purposes. Please:

- Do not use for impersonation without explicit consent
- Do not create misleading or deceptive content
- Follow all applicable laws and ethical guidelines regarding synthetic media

## 📄 License

The Gradio interface is licensed under the Apache 2.0 License. The CSM-1B model has its own license terms available at [Hugging Face](https://huggingface.co/sesame/csm-1b).

## 🙏 Acknowledgements

- [Sesame AI Labs](https://www.sesame.com/) for creating and open-sourcing CSM-1B
- [Hugging Face](https://huggingface.co/) for hosting the model
- [Gradio](https://gradio.app/) for the web interface framework