docker run -it --name mongodb \
    --privileged=true --restart=always \
    -e MONGO_INITDB_ROOT_USERNAME=admin \
    -e MONGO_INITDB_ROOT_PASSWORD=admin \
    -p 27017:27017 \
    mongo:latest