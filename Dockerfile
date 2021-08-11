FROM python:3.7
COPY . /app
RUN pip install -r /app/requirements.txt && pip install "pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"

