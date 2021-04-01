#!/bin/bash

docker build -t nova/cnn .
docker run docker.io/nova/cnn
