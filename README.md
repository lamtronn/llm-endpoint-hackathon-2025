# DICOM Image Analysis API

A FastAPI server that processes DICOM medical images using BLIP2 image-to-text model.

## 🎯 Purpose

This API accepts a URL to a DICOM file, downloads it, converts it to JPG→Base64, and processes it with BLIP2 model to generate image descriptions.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Server

```bash
python main.py
```

The server will:
- Load BLIP2 model (~2.7GB, first run downloads from HuggingFace)
- Start on `http://localhost:8000`

## 📡 API Endpoint

### POST /process-dicom-url

Process a DICOM file from URL.

**Request:**
```json
{
  "dicom_url": "https://example.com/scan.dcm",
  "prompt": "Describe this medical image"
}
```

**Response:**
```json
{
  "status": "success",
  "dicom_url": "https://example.com/scan.dcm",
  "dicom_metadata": {
    "PatientID": "12345",
    "StudyDate": "20241124",
    "Modality": "CT",
    "ImageSize": "512x512"
  },
  "analysis": "a medical scan showing anatomical structures",
  "model_used": "blip2-opt-2.7b",
  "processing_steps": {
    "1_downloaded": "DICOM file downloaded to server",
    "2_converted_to_jpg": "JPG size: 234567 bytes",
    "3_converted_to_base64": "Base64 length: 313424 chars",
    "4_sent_to_model": "BLIP2 processed the image"
  }
}
```

## 📝 Usage Examples

### curl
```bash
curl -X POST http://localhost:8000/process-dicom-url \
  -H "Content-Type: application/json" \
  -d '{
    "dicom_url": "https://your-server.com/scan.dcm"
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/process-dicom-url",
    json={
        "dicom_url": "https://your-server.com/scan.dcm",
        "prompt": "Describe this image"
    }
)

print(response.json())
```

### Postman
1. **Method:** POST
2. **URL:** `http://localhost:8000/process-dicom-url`
3. **Headers:** `Content-Type: application/json`
4. **Body (raw JSON):**
   ```json
   {
     "dicom_url": "https://your-server.com/scan.dcm"
   }
   ```

## 🔄 Processing Pipeline

```
1. DICOM URL → Download to server temp file
2. Read DICOM → Extract pixel array
3. Normalize pixels → Convert to PIL Image
4. Save as JPG in memory
5. Encode JPG to Base64
6. Decode back to PIL Image
7. Process with BLIP2 model
8. Return analysis + metadata
9. Clean up temp files
```

## 🛠️ Tech Stack

- **FastAPI** - REST API framework
- **BLIP2** - Salesforce/blip2-opt-2.7b (image-to-text)
- **PyDICOM** - DICOM file handling
- **Pillow** - Image processing
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model library

## 📦 Dependencies

See [requirements.txt](requirements.txt):
- fastapi
- uvicorn
- transformers
- torch
- pydicom
- Pillow
- numpy
- accelerate

## 🔧 Configuration

Create a `.env` file (optional):
```env
MODEL_NAME=blip2-opt-2.7b
API_URL=http://localhost:8000
```

## 🌐 API Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## ⚙️ System Requirements

### Minimum:
- Python 3.8+
- 8GB RAM
- 10GB disk space (for model)

### Recommended:
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with CUDA (for faster inference)
- 20GB disk space

## 🎯 Model Information

**BLIP2-OPT-2.7B:**
- Type: Vision-Language Model
- Task: Image Captioning
- Size: ~2.7B parameters
- Speed:
  - CPU: 2-5 seconds per image
  - GPU: 0.5-1 second per image

## 🔒 Security Considerations

Before production deployment:

1. **Add authentication** (API keys, OAuth)
2. **Validate URLs** (prevent SSRF attacks)
3. **Limit file sizes** (prevent DoS)
4. **Rate limiting** (protect resources)
5. **HTTPS only** (encrypt traffic)
6. **Logging** (audit trail)

## 📊 Monitoring

Server logs show:
```
Loading BLIP2 model...
BLIP2 model loaded on device: cpu
INFO:     Started server process
Downloading DICOM from: https://...
DICOM downloaded to: /tmp/tmpXXX.dcm
DICOM converted to image: (512, 512)
Image converted to JPG: 234567 bytes
JPG converted to base64: 313424 chars
Processing with BLIP2 model...
Cleaned up temporary DICOM file
```

## 🐛 Troubleshooting

### Model not downloading?
```bash
# Manual download
python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b'); Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')"
```

### Out of memory?
- Use smaller batch sizes
- Enable 8-bit quantization
- Run on CPU (slower but uses less RAM)

### DICOM download fails?
- Check URL is accessible
- Verify network connectivity
- Check firewall settings

## 📄 License

MIT

## 🤝 Contributing

Issues and PRs welcome!

## 📚 Documentation

- [Requirements Verification](REQUIREMENTS_VERIFICATION.md)
- [BLIP2 Setup Guide](BLIP2_SETUP.md)
# llm-endpoint-hackathon-2025
