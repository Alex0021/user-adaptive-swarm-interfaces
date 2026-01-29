import time
import debugpy
import os
import zmq

def main():

    if os.environ.get("PYTHONDEBUG", "0") == "1":
        print("Waiting for debugger to attach...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    if os.environ.get("IS_DOCKER", "0") == "1":
        socket.connect("tcp://host.docker.internal:5555")
    else:
        socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        data = socket.recv_json()
        print(f"Received data: {data}\n")
        time.sleep(0.01)

if __name__ == "__main__":
    main()