FROM python:3.10

ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt && python3 -m spacy download en_core_web_sm

CMD ["python", "main.py"]
