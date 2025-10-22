# Modal.com 
```python
# If you don't have a token yet
uv run modal token new

uv run modal serve main.py
```

# Local
```python
uv run python server.py

# Or with hot-reload
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

# Example request body 
```json
{
    "request": {"format": "mp4", "id": "test"},
    "source": {"url":"https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_5MB.mp4"}
}
```
