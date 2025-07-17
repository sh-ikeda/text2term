FROM python:3.9.23-alpine3.22

RUN apk update && \
    apk add --no-cache gcc libc-dev g++ build-base linux-headers

RUN mkdir /app
COPY . /app/text2term
WORKDIR /app/text2term

RUN pip install .

ENV PYTHONPATH=/app/text2term/text2term/

ENTRYPOINT ["python3", "-m", "text2term"]
CMD ["-h"]
