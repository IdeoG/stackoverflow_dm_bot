# Stackoverflow question search chatbot

Final assignment of Advanced Machine Learning NLP HSE Course

## How to run?

```bash
docker build -t stackoverflow_bot .
docker run --rm -d stackoverflow_bot python3 main_bot.py --token <your_bot_token>
```