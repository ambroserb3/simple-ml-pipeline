FROM python:3.8

WORKDIR /app
RUN mkdir /model
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN ls
COPY . .

CMD ["python", "train.py"]
