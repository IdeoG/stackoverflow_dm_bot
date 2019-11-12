FROM python:3.5
COPY . /root/app
WORKDIR /root/app
RUN pip install -r requirements.txt
CMD bash