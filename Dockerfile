#FROM stablebaselines/rl-baselines3-zoo:0.11.0a4
FROM star16231108/mybaseline:1.0
COPY . /app
WORKDIR /app
#RUN pip install -r /app/requirements.txt && pip install "pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"

