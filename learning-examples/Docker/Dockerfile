# We first start with a base image. Multiple bases are available on Docker Hub.
FROM node:alpine

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# CMD specifies what command to run within the container. As we set the working directory to /app,
# we can run app.js directly with node.
CMD node app.js

# To build the Docker image in this repository, run:
# docker build -t hello-docker .