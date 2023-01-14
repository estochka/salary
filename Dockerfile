FROM python:3.10

ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
