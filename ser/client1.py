import socket
import base64
import time

def receive_image(client_socket, save_path):
    i = 0
    while True:
        save_path1 = save_path + str(i) + ".png"
        with open(save_path1, 'wb') as file:
            if i >0:
                data_c = "1"
                client_socket.sendall(data_c.encode('utf-8'))
            data = client_socket.recv(4096)
            print(data)
            if not data:
                print("Failed to read")
                break
            else:
                file.write(base64.b64decode(data))
                print('Image received and saved.')
                time.sleep(5)
        client_socket.close()
    

def main():
    server_address = 'localhost'  # 替换为服务器地址
    server_port = 8005  # 替换为服务器端口号
    save_path = 'received_image'  # 图片保存路径
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, server_port))
    data = "1"
    client_socket.sendall(data.encode('utf-8'))
    # print(data.encode('utf-8'))
    # receive_image(client_socket, save_path)
    i = 0
    while True:
        save_path1 = save_path + str(i) + ".jpg"
        with open(save_path1, 'wb') as file:
            if i > 0:
                data_c = "1"
                client_socket.sendall(data_c.encode('utf-8'))
            data = client_socket.recv(409600)
            print(data)
            if not data:
                print("Failed to read")
                break
            else:
                file.write(base64.b64decode(data))
                print(base64.b64decode(data))
                print('Image received and saved.')
                time.sleep(3)
                i = i + 1
    client_socket.close()
    
    # receive_image(server_address, server_port, save_path)
   
if __name__ == '__main__':
    main()