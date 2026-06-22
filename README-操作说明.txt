========================================
命令区别：
  up      → 创建并启动容器（如果不存在则创建，存在则启动）
  down    → 停止并删除容器（容器没了，数据卷还在）
  start   → 启动已存在的已停止容器（不创建）
  stop    → 停止正在运行的容器（不删除）
========================================

========== docker-compose.yml（基础设施：mysql / mongodb / es / etcd / minio / milvus）==========

启动全部： docker-compose -f "docker-compose.yml" up -d
停止全部： docker-compose -f "docker-compose.yml" down
启动单个： docker-compose -f "docker-compose.yml" up -d mysql
停止单个： docker-compose -f "docker-compose.yml" stop mysql
启动单个（容器名）： docker start rag-mysql
停止单个（容器名）： docker stop rag-mysql

========== docker-compose-vis.yml（可视化：phpmyadmin / mongo-express / elasticvue / attu）==========

启动全部： docker-compose -f "docker-compose-vis.yml" up -d
停止全部： docker-compose -f "docker-compose-vis.yml" down
启动单个： docker-compose -f "docker-compose-vis.yml" up -d phpmyadmin
停止单个： docker-compose -f "docker-compose-vis.yml" stop phpmyadmin
启动单个（容器名）： docker start phpmyadmin
停止单个（容器名）： docker stop phpmyadmin



这个网络有什么作用
+ Thought: 1.7s
Docker 默认的网络隔离机制——不同 docker-compose.yml 启动的容器，默认在不同的网络里，互相看不到对方的容器名。
rag_default 就是让跨项目的容器（比如 ResumeRAG 的 rag-mysql 和 SmartQuery 的 sq-phpmyadmin）能通过容器名直接通信，不然 sq-phpmyadmin 就找不到 rag-mysql 这个地址。



两个项目用同一套配置，一个服务器端口号只能连一个端口号，数据库，可视化都一样，放不同表里面就行了