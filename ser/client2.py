import socket
import base64
import json
import time
def receive_image(server_address, port, save_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    client_socket.connect((server_address, port))
    data_c = "1"
    client_socket.sendall(data_c.encode('utf-8'))
    buffer = b''
    while True:
        data = client_socket.recv(4096)
        print(data)
        if not data:
            print("no data available")
            break
        buffer += data
    print("received over")
    json_data = buffer.decode('utf-8')
    data = json.loads(json_data)
    base64_data = data["image"].encode('utf-8')
    image_data = base64.b64decode(base64_data)
    text_name = data["text_name"]
    print(text_name)
    text_prob = data["text_prob"]
    print(text_prob)
    text_level = data["text_level"]
    print(text_level)
    with open(save_path, 'wb') as file:
        file.write(image_data)

    client_socket.close()
    
    print('Image received and saved.')

def main():
    server_address = 'localhost'  # 替换为服务器地址
    server_port = 8001  # 替换为服务器端口号
    save_path = './image/a'  # 图片保存路径
    i = 0
    while True:
        save_img_path = save_path + str(i) + '.jpg'
        receive_image(server_address, server_port, save_img_path)
        time.sleep(1)
        i = i + 1

if __name__ == '__main__':
    main()