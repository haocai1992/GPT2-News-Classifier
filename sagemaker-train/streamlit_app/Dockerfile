# base image
FROM python:3.7.4-slim-stretch

# Docker argument
ARG SAGEMAKER_ENDPOINT_NAME
ENV SAGEMAKER_ENDPOINT_NAME=$SAGEMAKER_ENDPOINT_NAME

# exposing default port for streamlit
EXPOSE 80
# EXPOSE 8501

# making directory of app
# WORKDIR /usr/src/app
WORKDIR /streamlit-gpt2-news-classifier-app

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copying all files over
# COPY src/ src/
COPY . .

# cmd to launch app when container is run
# ENTRYPOINT [ "streamlit", "run", "src/app.py", \
#              "--browser.serverAddress", "0.0.0.0", \
#              "--server.enableCORS", "false", \
#              "--server.enableXsrfProtection", "false",\
#              "--server.port", "80"]
CMD streamlit run ./src/app.py --server.port=80

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml'