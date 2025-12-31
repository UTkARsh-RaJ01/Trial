# LangGraph Model Server

A FastAPI-based server for managing and executing LangGraph models.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:3000`

## API Endpoints

### Base URL
- `GET /`: Check if the server is running

### Model Management
- `GET /models`: List all registered models
- `POST /models`: Create a new model
  ```json
  {
    "name": "model_name",
    "config": {
      // model-specific configuration
    }
  }
  ```
- `DELETE /models/{model_name}`: Delete a model
- `POST /models/{model_name}/execute`: Execute a model with input data
  ```json
  {
    // model-specific input data
  }
  ```

## Adding New Models

To add a new LangGraph model:

1. Create a new file in the `models` directory
2. Implement your model using the LangGraph framework
3. Export a function that creates and returns your model
4. Update the main server to use your model

## Example Usage

```python
import requests

# Create a new model
response = requests.post("http://localhost:3000/models", json={
    "name": "example_model",
    "config": {}
})

# Execute the model
response = requests.post("http://localhost:3000/models/example_model/execute", json={
    "input": "Hello, World!"
})
```

## Development

The server uses FastAPI's automatic API documentation. You can access it at:
- Swagger UI: `http://localhost:3000/docs`
- ReDoc: `http://localhost:3000/redoc` 