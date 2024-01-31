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
    echo -e "++++ Iteration: ${i} ++++"
    #from manpage of mkfifo: 
    
    #However, it has to be open at both ends simultaneously before you can proceed to do any input or output operations on it. 
    #Opening a FIFO for reading normally blocks until some other process opens the same FIFO for writing, and vice versa. See fifo(7) for nonblocking handling of FIFO special files. 
    python use_pipe.py pipe=${PIPE_NAME} run=create text=${text} args=$args &
    python use_pipe.py pipe=${PIPE_NAME} run=consume
    echo
done
rm -f ${PIPE_NAME}