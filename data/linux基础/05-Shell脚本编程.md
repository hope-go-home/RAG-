# 第 5 章：Shell 脚本编程

## 5.1 什么是 Shell 脚本？

Shell 脚本就是**把一堆命令写进文件里，一次性执行**。

你手工操作：
```bash
cd ~/project
mkdir -p src include build
touch src/main.c include/header.h
echo "项目已创建"
```

写成脚本 `init_project.sh`：
```bash
#!/bin/bash
cd ~/project
mkdir -p src include build
touch src/main.c include/header.h
echo "项目已创建"
```

执行：`bash init_project.sh` 或 `./init_project.sh`

### 为什么要学 Shell 脚本？

| 场景 | 手工操作 | 脚本自动化 |
|---|---|---|
| 部署服务器 | 敲 50 个命令 | 运行 1 个脚本 |
| 备份文件 | 每天手动操作 | 定时任务自动执行 |
| 批量处理 | 重复 1000 次 | 脚本循环处理 |
| 日志分析 | 一个个文件查看 | 脚本自动分析 |

## 5.2 第一个脚本

### Shebang

```bash
#!/bin/bash
# ↑ 这叫 shebang，告诉系统用哪个解释器执行
# 常见的 shebang：
#!/bin/bash         # Bash
#!/bin/sh           # Bourne Shell（最可移植）
#!/usr/bin/python3  # Python
#!/usr/bin/env node # Node.js
```

### 执行方式

```bash
# 方式 1：作为 bash 参数
bash script.sh

# 方式 2：加执行权限后直接执行
chmod +x script.sh
./script.sh

# 方式 3：source 执行（在当前 Shell 中运行）
source script.sh
. script.sh           # 和 source 相同
```

**区别**：
- `bash script.sh` 和 `./script.sh`：创建**子 Shell**执行
- `source script.sh`：在**当前 Shell**中执行（可以改变环境变量）

## 5.3 ⚡ 变量

### 变量定义和使用

```bash
#!/bin/bash

# 定义变量（等号两边不能有空格）
name="Alice"
age=25

# 使用变量（加 $）
echo $name
echo "My name is $name"     # 双引号中解析变量
echo 'My name is $name'     # 单引号中不解析，原样输出
echo ${name}                # 花括号可省略，但建议加（避免歧义）

# 只读变量
readonly PI=3.14159
# PI=3.14  # 错误！只读变量不能修改

# 删除变量
unset age
echo $age                    # 输出空
```

### 变量类型

Shell 变量**都是字符串**（不做类型区分）：
```bash
a=10
b=20
echo $a + $b         # 输出 "10 + 20"（字符串拼接）
echo $((a + b))      # 输出 30（需要 $(( )) 做算术运算）
```

### 环境变量

```bash
# 查看环境变量
echo $PATH
echo $HOME
echo $SHELL
env                    # 列出所有环境变量

# 设置环境变量
export MY_VAR="hello"  # 导出为环境变量（子进程可以继承）
MY_VAR="hello"         # 不导出（只在当前 Shell 有效）
```

### 特殊变量

```bash
#!/bin/bash

echo "脚本名称: $0"
echo "参数个数: $#"
echo "所有参数: $@"
echo "所有参数: $*"
echo "第一个参数: $1"
echo "第二个参数: $2"
echo "第十个参数: ${10}"       # 两位数要加花括号
echo "当前进程 PID: $$"
echo "上条命令退出码: $?"      # 0=成功，非0=失败
echo "上条后台命令 PID: $!"
```

**$@ vs $***：
```bash
# 假设脚本接收参数 a b c
# $@ → "a" "b" "c"（每个是独立参数）
# $* → "a b c"（整体一个参数）

for arg in "$@"; do
    echo "参数: $arg"
done
```

## 5.4 字符串操作

```bash
#!/bin/bash

str="Hello, World!"

# 字符串长度
echo ${#str}              # 13

# 切片
echo ${str:0:5}           # Hello（从0开始，取5个）
echo ${str:7}             # World!（从7开始到末尾）
echo ${str:(-6)}          # World!（从倒数第6到末尾）

# 替换
echo ${str/Hello/Hi}      # Hi, World!（替换第一个）
echo ${str//o/O}          # HellO, WOrld!（替换全部）
echo ${str/#Hello/Hi}     # 行首匹配替换
echo ${str/%,/!}          # Hello! World!（行尾匹配替换）

# 删除匹配前缀/后缀
filename="photo_2024.jpg"
echo ${filename#photo_}   # 2024.jpg（删除最短前缀）
echo ${filename##*_}      # 2024.jpg（删除最长前缀）
echo ${filename%.jpg}     # photo_2024（删除最短后缀）
echo ${filename%%.*}      # photo_2024（删除最长后缀）

# 默认值
echo ${name:-"无名"}       # 如果 name 未定义，使用默认值
echo ${name:="无名"}       # 如果 name 未定义，设置默认值
echo ${name:?"错误：name未定义"}  # 未定义则报错
```

## 5.5 ⚡ 条件判断

### test 命令和 [ ]

```bash
# 格式 1：test 条件
if test "$a" = "$b"; then
    echo "相等"
fi

# 格式 2：[ 条件 ]（注意空格！）
if [ "$a" = "$b" ]; then
    echo "相等"
fi

# 格式 3：[[ 条件 ]]（bash 扩展，更强大）
if [[ "$a" == "$b" ]]; then
    echo "相等"
fi
```

### 文件测试

```bash
#!/bin/bash

file="/etc/passwd"

[ -e "$file" ]  && echo "文件存在"       # -e: exist
[ -f "$file" ]  && echo "普通文件"       # -f: file
[ -d "$file" ]  && echo "目录"          # -d: directory
[ -L "$file" ]  && echo "符号链接"      # -L: link
[ -r "$file" ]  && echo "可读"          # -r: readable
[ -w "$file" ]  && echo "可写"          # -w: writable
[ -x "$file" ]  && echo "可执行"        # -x: executable
[ -s "$file" ]  && echo "非空文件"      # -s: size > 0
[ -N "$file" ]  && echo "已被修改过"    # -N: newer than last read
```

### 字符串比较

```bash
#!/bin/bash

str1="hello"
str2="world"

[ "$str1" = "$str2" ]    && echo "相等"        # POSIX 风格
[ "$str1" != "$str2" ]   && echo "不等"
[[ "$str1" == "$str2" ]] && echo "相等"        # bash 风格
[[ "$str1" < "$str2" ]]  && echo "小于"        # 字典序
[[ "$str1" > "$str2" ]]  && echo "大于"
[ -z "$empty" ]          && echo "空字符串"    # zero length
[ -n "$str1" ]           && echo "非空"        # non-zero
```

### 数字比较

```bash
#!/bin/bash

a=10
b=20

[ "$a" -eq "$b" ] && echo "等于"      # equal
[ "$a" -ne "$b" ] && echo "不等于"    # not equal
[ "$a" -gt "$b" ] && echo "大于"      # greater than
[ "$a" -ge "$b" ] && echo "大于等于"  # greater or equal
[ "$a" -lt "$b" ] && echo "小于"      # less than
[ "$a" -le "$b" ] && echo "小于等于"  # less or equal

# 也可以用 (( ))（bash 风格）
((a > b))   && echo "大于"
((a + 10 > b)) && echo "a+10 > b"
```

### 逻辑运算符

```bash
#!/bin/bash

# 传统风格
[ "$a" -gt 0 -a "$b" -lt 100 ] && echo "a>0 且 b<100"    # -a = and
[ "$a" -gt 0 -o "$b" -lt 100 ] && echo "a>0 或 b<100"    # -o = or
[ ! -z "$str" ]                 && echo "非空"            # ! = not

# bash 风格（推荐）
[[ $a -gt 0 && $b -lt 100 ]] && echo "a>0 且 b<100"
[[ $a -gt 0 || $b -lt 100 ]] && echo "a>0 或 b<100"
[[ ! -z $str ]]              && echo "非空"
```

### if-elif-else

```bash
#!/bin/bash

score=$1

if [ -z "$score" ]; then
    echo "请输入分数"
    exit 1
fi

if [ "$score" -ge 90 ]; then
    echo "优秀"
elif [ "$score" -ge 80 ]; then
    echo "良好"
elif [ "$score" -ge 60 ]; then
    echo "及格"
else
    echo "不及格"
fi
```

### case 语句

```bash
#!/bin/bash

echo "选择操作："
echo "1) 启动"
echo "2) 停止"
echo "3) 重启"
read choice

case "$choice" in
    1|start|Start)
        echo "启动服务..."
        systemctl start nginx
        ;;
    2|stop|Stop)
        echo "停止服务..."
        systemctl stop nginx
        ;;
    3|restart|Restart)
        echo "重启服务..."
        systemctl restart nginx
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
```

## 5.6 ⚡ 循环

### for 循环

```bash
#!/bin/bash

# 遍历列表
for fruit in apple banana orange; do
    echo "I like $fruit"
done

# 遍历数字序列
for i in {1..5}; do
    echo "数字: $i"
done

# C 风格 for
for ((i=0; i<5; i++)); do
    echo "计数: $i"
done

# 遍历文件
for file in *.txt; do
    echo "处理: $file"
    wc -l "$file"
done

# 遍历命令输出
for user in $(cat /etc/passwd | cut -d: -f1); do
    echo "用户: $user"
done
```

### while 循环

```bash
#!/bin/bash

# 计数循环
count=1
while [ "$count" -le 5 ]; do
    echo "第 $count 次"
    count=$((count + 1))
done

# 读取文件
while IFS= read -r line; do
    echo "行: $line"
done < /etc/passwd          # 输入重定向

# 无限循环（用 Ctrl+C 退出）
while true; do
    echo "服务器运行中..."
    sleep 1
done

# 监控进程
while pgrep -x "nginx" > /dev/null; do
    echo "nginx 正在运行..."
    sleep 5
done
echo "nginx 已停止"
```

### until 循环

```bash
#!/bin/bash

# until = while 的反面（条件为假时执行）
count=10
until [ "$count" -lt 1 ]; do
    echo "倒计时: $count"
    count=$((count - 1))
    sleep 1
done
echo "时间到！"
```

### break 和 continue

```bash
#!/bin/bash

# break — 跳出循环
for i in {1..10}; do
    if [ "$i" -eq 5 ]; then
        break           # i=5 时跳出循环
    fi
    echo $i
done

# continue — 跳过本次循环
for i in {1..10}; do
    if [ "$((i % 2))" -eq 0 ]; then
        continue        # 偶数跳过
    fi
    echo $i             # 只输出奇数
done
```

## 5.7 ⚡ 函数

```bash
#!/bin/bash

# 定义函数
say_hello() {
    echo "Hello, $1!"    # $1 = 第一个参数
}

# 调用
say_hello "Alice"
say_hello "Bob"

# 带返回值的函数
add() {
    local a=$1          # local = 局部变量
    local b=$2
    echo $((a + b))     # 用 echo 返回结果
    return 0            # return 只能返回 0-255 的状态码
}

result=$(add 10 20)     # 捕获输出
echo "10 + 20 = $result"

# 检查文件是否存在
check_file() {
    if [ -f "$1" ]; then
        echo "文件存在"
        return 0
    else
        echo "文件不存在"
        return 1
    fi
}

if check_file "/etc/passwd"; then
    echo "检查通过"
fi

# 函数中修改全局变量
global_var="外面"
modify() {
    global_var="里面被改了"    # 默认修改的是全局变量
    local local_var="局部"    # local 关键字声明局部变量
}
modify
echo "$global_var"    # 输出 "里面被改了"
```

## 5.8 ⚡ 输入输出重定向深入

### 文件描述符操作

```bash
# 标准用法
command > file      # stdout → file
command 2> file     # stderr → file
command &> file     # stdout+stderr → file
command >> file     # 追加 stdout
command 2>&1        # stderr → stdout（合并到 stdout）

# 深入理解 2>&1
echo "test" > file 2>&1
# 解析：
# 1. > file — 将 stdout (fd 1) 重定向到 file
# 2. 2>&1  — 将 stderr (fd 2) 重定向到 stdout (fd 1)
# 注意顺序！

# 理解错误顺序
echo "test" 2>&1 > file
# 1. 2>&1  — stderr → stdout（当前 stdout 还是屏幕）
# 2. > file — stdout → file（但 stderr 仍然指向屏幕）

# 自定义文件描述符
exec 3> output.txt      # 打开 fd 3 用于写入文件
echo "到文件" >&3       # 写入 fd 3
exec 3>&-               # 关闭 fd 3

# 读取文件到 fd
exec 4< input.txt       # 打开 fd 4 读取文件
read -u 4 line          # 从 fd 4 读取一行
exec 4<&-               # 关闭 fd 4
```

### Here Document

```bash
#!/bin/bash

# 生成多行文本
cat << EOF
这是第一行
这是第二行
这是第三行
EOF

# 重定向到文件
cat << EOF > config.txt
server {
    listen 80;
    server_name example.com;
}
EOF

# 变量替换
name="Alice"
cat << EOF
Hello, $name
EOF

# 不进行变量替换（引号括起来）
cat << 'EOF'
Hello, $name    # 原样输出 $name
EOF
```

## 5.9 ⚡ 数组

```bash
#!/bin/bash

# 定义数组
fruits=("apple" "banana" "orange")
numbers=(1 2 3 4 5)

# 访问元素
echo ${fruits[0]}       # apple
echo ${fruits[1]}       # banana
echo ${fruits[@]}       # 所有元素
echo ${fruits[*]}       # 所有元素

# 数组长度
echo ${#fruits[@]}      # 3

# 追加元素
fruits+=("grape")
fruits[5]="watermelon"  # 指定索引

# 遍历数组
for fruit in "${fruits[@]}"; do
    echo "$fruit"
done

# 切片
echo "${fruits[@]:0:2}"  # apple banana（从0开始，取2个）

# 关联数组（类似字典/哈希表）
declare -A person
person["name"]="Alice"
person["age"]=25
person["city"]="Beijing"

echo ${person["name"]}    # Alice
echo ${!person[@]}        # 所有键：name age city
```

## 5.10 实用脚本模式

### 脚本模板

```bash
#!/bin/bash
#
# 脚本名称: example.sh
# 描述: 这是一个脚本模板
# 作者: Alice
# 用法: ./example.sh [选项] [参数]

set -euo pipefail   # 严格模式
IFS=$'\n\t'         # 安全的 IFS

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'        # No Color

# 日志函数
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 使用说明
usage() {
    cat << EOF
用法: $0 [选项] <文件名>

选项:
    -h, --help      显示帮助信息
    -v, --verbose   详细输出
    -o, --output    输出文件

示例:
    $0 -o result.txt input.txt
EOF
    exit 1
}

# 参数解析
verbose=0
output_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            verbose=1
            shift
            ;;
        -o|--output)
            output_file="$2"
            shift 2
            ;;
        -*)
            log_error "未知选项: $1"
            usage
            ;;
        *)
            input_file="$1"
            shift
            ;;
    esac
done

# 参数检查
if [ -z "${input_file:-}" ]; then
    log_error "缺少输入文件"
    usage
fi

if [ ! -f "$input_file" ]; then
    log_error "文件不存在: $input_file"
    exit 1
fi

# 主逻辑
log_info "处理文件: $input_file"
[ "$verbose" -eq 1 ] && log_info "详细输出模式开启"

# 处理...
log_info "完成！"
```

### set -euo pipefail 详解

```bash
set -e    # 脚本中任何命令失败立即退出（error on）
          # 防止错误蔓延

set -u    # 使用未定义变量时报错（nounset）
          # 而不是当作空字符串

set -o pipefail  # 管道中任一步骤失败，整个管道返回非0
                 # 默认只返回最后一步的结果

# 没有 set -e 时：
false
echo "这行还是会执行"    # 错误被忽略！

# 有 set -e 时：
set -e
false
echo "这行不会执行"     # 脚本在这里退出

# pipefail 示例：
set -o pipefail
true | false | true
echo $?    # 输出 1（中间的 false 导致管道失败）
           # 没有 pipefail 会输出 0（只看最后的 true）
```

## 5.11 调试技巧

```bash
#!/bin/bash

# 方式 1：bash -x 执行（显示每条命令及其展开）
bash -x script.sh

# 方式 2：脚本中启用
set -x    # 开始调试
# ... 要调试的代码 ...
set +x    # 关闭调试

# 方式 3：bash -n 检查语法（不执行）
bash -n script.sh

# 方式 4：bash -v 显示原始行
bash -v script.sh

# PS4 自定义调试提示符
export PS4='+${BASH_SOURCE}:${LINENO}:${FUNCNAME[0]}: '
bash -x script.sh
# 输出：+myscript.sh:10:main: echo "hello"
```

## 💡 本章思考题

1. `$@` 和 `$*` 在加引号时有什么区别？
2. `[ ]` 和 `[[ ]]` 有什么区别？
3. `2>&1` 和 `&>` 写法等价吗？
4. `set -e` 有什么潜在陷阱？
5. `source script.sh` 和 `./script.sh` 有什么区别？

## 🔧 动手练习

```bash
# 1. 创建一个备份脚本
cat > ~/backup.sh << 'EOF'
#!/bin/bash
# 备份家目录中的文档
backup_dir="$HOME/backups/$(date +%Y%m%d)"
mkdir -p "$backup_dir"
cp -r "$HOME/Documents" "$backup_dir/"
echo "备份完成: $backup_dir"
EOF

chmod +x ~/backup.sh
./backup.sh

# 2. 创建批量重命名脚本
cat > ~/rename.sh << 'EOF'
#!/bin/bash
# 将所有 .txt 改为 .md
for file in *.txt; do
    if [ -f "$file" ]; then
        mv "$file" "${file%.txt}.md"
        echo "重命名: $file → ${file%.txt}.md"
    fi
done
EOF

# 3. 创建系统信息脚本
cat > ~/sysinfo.sh << 'EOF'
#!/bin/bash
echo "=== 系统信息 ==="
echo "主机名: $(hostname)"
echo "内核: $(uname -r)"
echo "CPU: $(nproc) 核"
echo "内存: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "磁盘: $(df -h / | awk 'NR==2 {print $3 "/" $2}')"
echo "IP: $(ip addr show | grep 'inet ' | awk '{print $2}' | head -1)"
echo "运行时间: $(uptime -p)"
echo "负载: $(uptime | awk -F'load average:' '{print $2}')"
EOF

chmod +x ~/sysinfo.sh
~/sysinfo.sh
```
