FROM python:3.7

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

# Get the arg from the build command with a default value
ARG LWLL_SESSION_MODE
ARG AWS_REGION_NAME
ARG DATASETS_BUCKET
ARG GOVTEAM_SECRET

# Set to an env variable
ENV LWLL_SESSION_MODE=$LWLL_SESSION_MODE
ENV AWS_REGION_NAME=$AWS_REGION_NAME
ENV DATASETS_BUCKET=$DATASETS_BUCKET
ENV GOVTEAM_SECRET=$GOVTEAM_SECRET

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]