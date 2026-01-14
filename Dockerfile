# syntax=docker/dockerfile:1.2
FROM python:latest as runtime-environment


COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U "pip>=21.2,<23.2"
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add composer user
ARG COMPOSER_UID=999
ARG COMPOSER_GID=0
RUN groupadd -f -g ${COMPOSER_GID} composer_group && \
useradd -m -d /home/composer_docker -s /bin/bash -g ${COMPOSER_GID} -u ${COMPOSER_UID} composer_docker

WORKDIR /home/composer_docker
USER composer_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG COMPOSER_UID=999
ARG COMPOSER_GID=0
COPY --chown=${COMPOSER_UID}:${COMPOSER_GID} challenge/api.py .

EXPOSE 8080

CMD ["uvicorn", "api.app", "runserver", "0.0.0.0:8080"]