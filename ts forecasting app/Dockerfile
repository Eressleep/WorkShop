FROM python:3.8-buster

RUN pip install --no-cache-dir fastapi uvicorn

COPY ./app .

ENTRYPOINT [ "uvicorn", "start:app", "--host", "0.0.0.0", "--port", "8000" ]