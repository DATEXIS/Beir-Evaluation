
#!/usr/bin/env bash

# build image
IMAGE=registry.datexis.com/mmenke/beir-eval

version=0.1.51
echo "Version: $version"
docker build -t $IMAGE -t $IMAGE:$version .
docker login -u $1 -p $2 d
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"
