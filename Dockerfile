FROM continuumio/anaconda3:4.4.0
COPY . /usr/app
EXPOSE 5000
WORKDIR /usr/app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --ignore-installed --upgrade pip setuptools
RUN pip install --upgrade tensorflow --ignore-installed
CMD python app.py