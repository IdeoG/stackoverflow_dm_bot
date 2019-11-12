FROM python:3.5
COPY . /root/app
ENV TOKEN="${TOKEN}"
WORKDIR /root/app
RUN pip install -r requirements.txt
CMD python3 main_bot.py --token TOKEN
