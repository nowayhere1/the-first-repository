import socket
import time

def receive_image(server_address, port, save_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, port))

    with open(save_path, 'wb') as file:
        while True:
            data = client_socket.recv(4096)
            # print(data)
            if not data:
                break
            file.write(data)
    client_socket.close()
    print('Image received and saved.')

def main():
    server_address = 'localhost'  # 替换为服务器地址
    server_port = 8001  # 替换为服务器端口号
    save_path = 'received_image.png'  # 图片保存路径
    while True:
        receive_image(server_address, server_port, save_path)
        time.sleep(2)

if __name__ == '__main__':
    main()
