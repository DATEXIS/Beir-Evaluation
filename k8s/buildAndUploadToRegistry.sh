#!/usr/bin/env bash

# build image
IMAGE=registry.datexis.com/mmenke/beir-eval

version=0.2.16
echo "Version: $version"
docker build -t $IMAGE -t $IMAGE:$version .
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"