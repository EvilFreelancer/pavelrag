FROM manticoresearch/manticore:6.3.8

ADD https://repo.manticoresearch.com/manticore-repo.noarch.deb /tmp/manticore-repo.noarch.deb

RUN set -xe  \
 && dpkg -i /tmp/manticore-repo.noarch.deb \
 && apt-get update \
 && apt-get install -fyq \
    manticore-columnar-lib \
 && rm -rf /var/lib/apt/lists/*
