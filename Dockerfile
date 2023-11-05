FROM python:3.10-slim

RUN apt-get update && \
    apt-get upgrade && \
    apt-get install -y --no-install-recommends libldap2-dev libsasl2-dev libssl-dev && \
    apt-get clean autoclean && rm -rf /var/lib/apt/* /var/cache/apt/* && \
    apt-get autoremove --purge && \
    pip install pipenv --no-cache-dir

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.pkl", "dv.pkl", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
