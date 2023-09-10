# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
                                                                   img.jpg                         # image
                                                                   vid.mp4                         # video
                                                                   screen                          # screenshot
                                                                   path/                           # directory
                                                                   list.txt                        # list of images
                                                                   list.streams                    # list of streams
                                                                   'path/*.jpg'                    # glob
                                                                   'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                   'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import re
import torch
import torch.nn.functional as F
import name_change as nc
import websockets
import asyncio
import base64
import socket
import json

# def send_image(client_socket, image_path, text_name,text_prob,text_jdown):   #发送图片
    # with open(image_path, 'rb') as file:
    #     # print(image_path)
    #     image_data = file.read()
    #     # # print(image_data)
    #     #         # 将图片数据转成Base64编码
    #     # base64_data = base64.b64encode(image_data)
    #     # client_socket.sendall(base64_data)
    #     # print(base64_data)
    #     # client_socket.sendall(image_data)
    #     base64_data = base64.b64encode(image_data)
    #     data = {
    #         "image": base64_data.decode('utf-8'),    # 图片
    #         "text_name": text_name,  # 地面类别
    #         "text_prob": text_prob,  # 置信度
    #         "text_level": text_jdown # 调平
    #     }
    #     json_data = json.dumps(data)
    #     # print(json_data)
    #     client_socket.sendall(json_data.encode('utf-8'))

async def send_image(websocket, path):
    print(path)
    await websocket.send("Processing images........")
    image_path,text_names,text_jdown,text_prob = run(**vars(opt))    # 图片路径
    await websocket.send("Images processing completed!")
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    img_leng = len(image_path)
    print(img_leng)
    i = 0
    # while True:
    #     # client_socket, client_address = server_socket.accept()
    #     # print('Accepted connection from', client_address)
    #     # cli_data = client_socket.recv(1024)
    #     # print("clidata",cli_data)
    #     cli_data = await websocket.recv()
    #     print(f"< {cli_data}")
    #     if not cli_data:
    #         break
    #     # elif cli_data == '1':
    #     # else:
    #     elif cli_data == "1":
    #     # image_path = '/home/ming/beili_websoket/test.png'  # 替换为你要传输的图片的路径
    #         # send_image(client_socket, img_path[i],text_names[i],text_prob[i],text_jdown[i])
    #         # send_text(i,text_names[i],text_prob[i],text_jdown[i])
    #         # print(i)
    #         with open(image_path[i], 'rb') as file:
    #         # print(image_path)
    #             image_data = file.read()
    #         # # print(image_data)
    #         #         # 将图片数据转成Base64编码
    #         # base64_data = base64.b64encode(image_data)
    #         # client_socket.sendall(base64_data)
    #         # print(base64_data)
    #         # client_socket.sendall(image_data)
    #             base64_data = base64.b64encode(image_data)
    #             data = {
    #                 "image": base64_data.decode('utf-8'),    # 图片
    #                 "text_name": text_names[i],  # 地面类别
    #                 "text_prob": text_prob[i],  # 置信度
    #                 "text_level": text_jdown[i] # 调平
    #             }
    #             json_data = json.dumps(data)
    #             # print(json_data)
    #             # await websocket.send(json_data.encode('utf-8'))
    #             await websocket.send(json_data)
    #             # print(json_data)
    #             # client_socket.sendall(json_data.encode('utf-8'))
    #             i = (i + 1) % img_leng
    #             print("第",i,"张")
    #             await asyncio.sleep(5)
    cli_data = await websocket.recv()
    print(f"< {cli_data}")
    # if not cli_data:
    #     break
    # elif cli_data == '1':
    # else:
    if cli_data == "1":
        while True:
            # client_socket, client_address = server_socket.accept()
            # print('Accepted connection from', client_address)
            # cli_data = client_socket.recv(1024)
            # print("clidata",cli_data)
            # cli_data = await websocket.recv()
            # print(f"< {cli_data}")
            # if not cli_data:
            #     break
            # # elif cli_data == '1':
            # # else:
            # elif cli_data == "1":
            # image_path = '/home/ming/beili_websoket/test.png'  # 替换为你要传输的图片的路径
                # send_image(client_socket, img_path[i],text_names[i],text_prob[i],text_jdown[i])
                # send_text(i,text_names[i],text_prob[i],text_jdown[i])
                # print(i)
                with open(image_path[i], 'rb') as file:
                # print(image_path)
                    image_data = file.read()
                # # print(image_data)
                #         # 将图片数据转成Base64编码
                # base64_data = base64.b64encode(image_data)
                # client_socket.sendall(base64_data)
                # print(base64_data)
                # client_socket.sendall(image_data)
                    base64_data = base64.b64encode(image_data)
                    data = {
                        "image": base64_data.decode('utf-8'),    # 图片
                        "text_name": text_names[i],  # 地面类别
                        "text_prob": text_prob[i],  # 置信度
                        "text_level": text_jdown[i] # 调平
                    }
                    json_data = json.dumps(data)
                    # print(json_data)
                    # await websocket.send(json_data.encode('utf-8'))
                    await websocket.send(json_data)
                    # print(json_data)
                    # client_socket.sendall(json_data.encode('utf-8'))
                    i = (i + 1) % img_leng
                    print("第",i,"张")
                    await asyncio.sleep(5)



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
# print("here",ROOT)   #D:\jupyter_notebook\yolov5-master\yolov5-master
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(224, 224),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,#False,  # show results
        save_txt=True,#False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-cls',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    
    # for i in range(27):
    #     print(str(i)+names[i]+'/n')
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # get accuracy
    getacc = 1
    acc_num = 0 #图像总量
    acc_right  = 0 #正确分类的数量
    acc = 0   #准确率
    # for i in range(27):
    #     print(str(i)+names[i]+'/n')
    names = ['不平整沥青路','平整沥青路','平整沥青路','不平整水泥路','平整水泥路','平整水泥路','砾石路','泥地',
             '雪地','冰面','雪地','不平整沥青路','平整沥青路','平整沥青路','不平整水泥路','平整水泥路','平整水泥路',
             '砾石路','泥地','不平整沥青路','平整沥青路','平整沥青路','不平整水泥路','平整水泥路','平整水泥路',
             '砾石路','泥地']
    
#############用于发送图片和识别结果
    img_path = []   # 存储原始图片的地址
    text_prob = []  # 存储置信度
    text_jdown = [] # 存储是否可以调平
    text_names = [] # 存储地形种类

    for path, im, im0s, vid_cap, s in dataset:  #路径、图像、原始图像的numpy、none、图片的宽和高
        
        img_path.append(path) #添加原始图片
        
        acc_num = acc_num + 1 
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities
        
        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            # img_path.append(save_path)    #添加图片名
            
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            txt_path = str(save_dir / 'labels' / "1") + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:1].tolist()  # top 3 indices   每一个类的索引
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "
            list1 = [1,2,4,5,12,13,15,16,20,21,23,24]
            # if top5i[0] in list1:
            #     jdown = "suitable"
            # else :
            #     jdown = "unsuitable"
            if top5i[0] in list1:
                jdown = "适合调平"
            else :
                jdown = "不适合调平"
                
            text_jdown.append(jdown)   #添加jdown
            text_names.append(names[top5i[0]])  #添加地形种类
            text_prob.append(round(prob[top5i[0]].item(),2))  #添加置信度
            print(names[top5i[0]])
            print(round(prob[top5i[0]].item(),2))
            print(jdown)
            # print()
            # print(top5i[0])
            # Write results
            text = '\n'.join(f'{prob[j]:.2f} {names[j]} \n{jdown}' for j in top5i)
            # print(text)
            # text = '\n'.join(f'{prob[j]:.2f} {nc.name_change(names[j])}' for j in top5i)
            # print(text)
            # # name0 = (f'{nc.name_change(names[0])}'  in top5i[1])
            # print()
            if save_img or view_img:  # Add bbox to image
                annotator.text((40, 40), text, txt_color=(0, 0, 255))
            if save_txt:  # Write to file
                with open(f'{txt_path}.txt', 'a') as f:
                # with open(f'{txt_path}.txt', 'a') as f:
                    folder_name = p.stem
                    pattern = re.compile(r"\d+-")
                    folder_name = pattern.sub("", folder_name, 1).replace('-', '_')
                    # 去掉扩展名，得到名称部分
                    folder_name = nc.name_change(folder_name)
                    f.write(text +" "+folder_name +" "+str(acc_right)+'\n')
                    if nc.name_change(names[top5i[0]]) ==  folder_name:
                        acc_right = acc_right + 1
          
            # Stream results
            im0 = annotator.result()
            if view_img:
                
                if platform.system() == 'Linux' and p not in windows:
                    
                    windows.append(p)
                    # cv2.namedWindow("img", 0);#cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # cv2.resizeWindow("img", im0.shape[1], im0.shape[0])
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # print("kankanzhili ",p)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                
                # cv2.namedWindow("img",0)
                # cv2.resizeWindow("img", 600, 800)
                im0 = cv2.resize(im0,(600,800))
                
                cv2.imshow("img", im0)
                # cv2.imshow(str(p), im0)
                cv2.waitKey(100)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        
        # Print time (inference-only)
        LOGGER.info(f'{s}\n time: {dt[1].dt * 1E3:.1f}ms')

    # Print results  输出照片的平均处理时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:       #输出日志信息
    #     print(0)
        # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print("s is:",s)
        # print("what")
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}") #将结果保存的提示信息打印到日志中
        # print("what is after")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    # if getacc:
        # print(acc_num,acc_right)
        # acc = "{:.2f}%".format(acc_right/acc_num*100)
        # print("accuracy is :",acc)
    return img_path,text_names,text_jdown,text_prob

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train-cls/exp4/weights/best.pt', help='model path(s)')   #27类
    parser.add_argument('--source', type=str, default=ROOT / 'data/1test', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')   #for docker
    # parser.add_argument('--source', type=str, default=ROOT / 'data/1test', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[224], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # img_path,text_names,text_jdown,text_prob = run(**vars(opt))    # 图片路径
    # check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # websoket()
    # path = 230
#########################9.8：0：30新添加的websocket部分
    host = socket.gethostname()
    print(host)
    start_server = websockets.serve(send_image, host, 8010)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
########################
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_address = ('localhost', 8001)
    # # host = socket.gethostname()
    # # server_address = (host, 8001)
    # server_socket.bind(server_address)
    # server_socket.listen(1)
    # print('Waiting for connections...')
    # client_socket, client_address = server_socket.accept()
    # print('Accepted connection from', client_address)
    # img_leng = len(img_path)
    # print(img_leng)
    # i = 0
    # while True:
    #     # client_socket, client_address = server_socket.accept()
    #     # print('Accepted connection from', client_address)
    #     # cli_data = client_socket.recv(1024)
    #     # print("clidata",cli_data)
    #     if not cli_data:
    #         break
    #     elif cli_data.decode('utf-8') == "1":
    #     # image_path = '/home/ming/beili_websoket/test.png'  # 替换为你要传输的图片的路径
    #         send_image(client_socket, img_path[i],text_names[i],text_prob[i],text_jdown[i])
    #         # send_text(i,text_names[i],text_prob[i],text_jdown[i])
    #         # print(i)
    #         i = (i + 1) % img_leng
    #         # if i < img_leng:
    #         #     i = (i + 1) % img_leng
    #         # else:
    #         #     break
    #         client_socket.close()

    # client_socket.close()
    # run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
