# base image
FROM python:3.7.4-slim-stretch

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /streamlit-gpt2-news-classifier-app

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# download model file
RUN apt-get update
RUN apt-get  -qq -y install wget
RUN wget -O ./model/gpt2-text-classifier-model.pt "https://www.dropbox.com/s/4m0i1t695qzvbfy/gpt2-text-classifier-model.pt?dl=1"

# cmd to launch app when container is run
CMD streamlit run ./src/app.py

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
" > /root/.streamlit/config.toml'
