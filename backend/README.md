# Ultimate Agents Hackathon Backend

A simple FastAPI backend application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /images` - Get 9 sample image URLs
- `POST /upload` - Upload an image (PNG or JPG)

## API Documentation

Once the server is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Example Usage

Get 9 image URLs:
```bash
curl -X GET "http://localhost:8000/images"
```

Upload an image:
```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@/path/to/your/image.jpg"
```

Upload an image (PNG):
```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@/path/to/your/image.png"
```
