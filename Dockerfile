FROM python:3.10-alpine

RUN mkdir /app
COPY . /app/text2term
WORKDIR /app/text2term

RUN pip install .

ENTRYPOINT ["python3", "-m", "text2term"]
CMD ["-h"]
