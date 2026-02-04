# Docker Compose Port Configuration Changes

## Date
February 4, 2026

## Summary
Updated local container port mappings in `docker/compose.yaml` to avoid port conflicts with other services.

## Changes Made

### Port Mappings Updated

| Service | Original Port | Updated Port | Purpose |
|---------|--------------|--------------|---------|
| mineru-openai-server | 30000 | 31001 | OpenAI-compatible API server |
| mineru-api | 8000 | 31002 | MinerU REST API |
| mineru-gradio | 7860 | 31003 | Gradio web interface |

## Technical Details

### File Modified
- `docker/compose.yaml`

### Changes by Service

#### 1. mineru-openai-server
```yaml
# Before
ports:
  - 30000:30000

# After
ports:
  - 31001:30000
```
- **Host Port**: Changed from `30000` to `31001`
- **Container Port**: Remains `30000` (internal)
- **Access**: `http://localhost:31001`

#### 2. mineru-api
```yaml
# Before
ports:
  - 8000:8000

# After
ports:
  - 31002:8000
```
- **Host Port**: Changed from `8000` to `31002`
- **Container Port**: Remains `8000` (internal)
- **Access**: `http://localhost:31002`

#### 3. mineru-gradio
```yaml
# Before
ports:
  - 7860:7860

# After
ports:
  - 31003:7860
```
- **Host Port**: Changed from `7860` to `31003`
- **Container Port**: Remains `7860` (internal)
- **Access**: `http://localhost:31003`

## Reason for Changes
- Avoid port conflicts with other commonly used services
- Port 8000 is frequently used by development servers
- Port 7860 is the default Gradio port and may conflict with other ML applications
- Port 30000 may conflict with other services
- The 31000 range provides a clean, sequential set of ports that are less commonly used

## Usage After Changes

### Starting Services
```bash
# Start OpenAI-compatible server
docker compose --profile openai-server up -d

# Start REST API
docker compose --profile api up -d

# Start Gradio interface
docker compose --profile gradio up -d
```

### Access URLs
- OpenAI Server: `http://localhost:31001`
- REST API: `http://localhost:31002`
- Gradio UI: `http://localhost:31003`

## Notes
- Only host-side ports were changed
- Internal container ports remain unchanged
- No application code changes required
- Docker Compose profiles remain the same
