import socket

def send_image(client_socket, image_path):
    with open(image_path, 'rb') as file:
        image_data = file.read()
        print(image_data)
        client_socket.sendall(image_data)

#服务端向客户端一次发送多个信息,包括图片和标注信息
def

def send_text(client_socket, text):
    client_socket.sendall(text.encode())

    

def main():
    #这是进行定义一个ip协议版本AF_INET（IPv4），定义一个传输TCP协议，SOCK_STREAM,括号里面包含两个参数，一个参数默认是ip地址蔟的socket.AF_INET，也就是IPv4；还有一个默认是传输TCP协议，也就是socket.SOCK_STREAM
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8001)  
    server_socket.bind(server_address)
    server_socket.listen(1)
    print('Waiting for connections...')
    client_socket, client_address = server_socket.accept()
    print('Accepted connection from', client_address)
    while True:
        cli_data = client_socket.recv(1024)
        if not cli_data:
            break
        elif cli_data.decode('utf-8') == "1":
            


        image_path = '/home/ming/beili_websoket/test.png'  # 替换为你要传输的图片的路径
        send_image(client_socket, image_path)

    client_socket.close()

if __name__ == '__main__':
    main()
