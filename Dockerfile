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

COPY ["app/.",  "./"]

COPY ["app/pages/.",  "./pages/"]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
