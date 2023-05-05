FROM python:3.8

WORKDIR /app
RUN mkdir /data

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m pytest test.py
COPY . .

CMD ["python", "prep.py"]
