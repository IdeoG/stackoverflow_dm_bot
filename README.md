# Stackoverflow question search chatbot

Final assignment of Advanced Machine Learning NLP HSE Course

## How to run?

```bash
docker build -t stackoverflow_bot .
docker run -e TOKEN=<YOUR_TOKEN> --rm -it stackoverflow_bot
```