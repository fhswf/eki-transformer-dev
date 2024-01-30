PIPE_NAME=$(mktemp -u)
mkfifo ${PIPE_NAME}
echo -e "\n--- Created pipe: ${PIPE_NAME} ---\n"
ITERATIONS=10

#first command writes something in pipe, second reads from it
for i in $(seq ${ITERATIONS}); do
    #random text
    text=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13)
    #writing needs to be in background
    args=$i
    python use_pipe.py pipe=${PIPE_NAME} run=create text=${text} args=$args &
    #right now, the output of the first create run script is fed into every consume script
    #TODO: fix this
    python use_pipe.py pipe=${PIPE_NAME} run=consume
done