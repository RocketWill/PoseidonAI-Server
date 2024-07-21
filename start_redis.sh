docker run -d --name redis \
    --privileged=true --restart=always \
    -p 6379:6379 \
    redis:6