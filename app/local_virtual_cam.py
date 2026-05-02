import cv2
import asyncio
import websockets
import numpy as np
import json
import struct
import argparse
import pyvirtualcam

# Para usar este script, instale as dependências no PC fraco:
# pip install opencv-python websockets pyvirtualcam numpy

async def virtual_cam_client(server_url, camera_index=0, out_device=None, width=640, height=480, fps=15):
    print(f"[*] Iniciando câmera local (Índice {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        print("[-] Erro ao abrir a webcam local.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"[+] Webcam pronta: {actual_width}x{actual_height} @ {actual_fps}fps")

    print(f"[*] Conectando ao servidor GPU em {server_url}...")
    try:
        async with websockets.connect(server_url) as ws:
            print("[+] Conectado ao servidor com sucesso!")
            
            # Inicializa a câmera virtual usando pyvirtualcam
            with pyvirtualcam.Camera(width=actual_width, height=actual_height, fps=actual_fps, fmt=pyvirtualcam.PixelFormat.BGR, device=out_device) as cam:
                print(f"[+] Câmera Virtual ativada: {cam.device}")
                print("[*] Transmitindo video. Para encerrar, pressione Ctrl+C no terminal.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[-] Erro ao ler frame da webcam.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 1. Envia frame para o servidor
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await ws.send(buffer.tobytes())
                    
                    # 2. Recebe frame processado
                    response = await ws.recv()
                    
                    if isinstance(response, bytes) and len(response) > 4:
                        json_len = struct.unpack("<I", response[:4])[0]
                        jpg_bytes = response[4+json_len:]
                        
                        np_arr = np.frombuffer(jpg_bytes, np.uint8)
                        processed_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if processed_frame is not None:
                            if processed_frame.shape[:2] != (actual_height, actual_width):
                                processed_frame = cv2.resize(processed_frame, (actual_width, actual_height))
                                
                            # Escreve o deepfake puro direto na Câmera Virtual (Dummy video device)
                            cam.send(processed_frame)
                            cam.sleep_until_next_frame()

    except Exception as e:
        print(f"[-] Erro na conexão: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Encerrado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Virtual Camera Client")
    parser.add_argument("--server", type=str, default="ws://192.168.18.254:8765", help="URL do servidor WebSocket GPU")
    parser.add_argument("--cam", type=int, default=0, help="Índice da webcam local")
    parser.add_argument("--out-cam", type=str, default=None, help="Caminho para a câmera virtual de saída")
    args = parser.parse_args()
    
    try:
        asyncio.run(virtual_cam_client(args.server, camera_index=args.cam, out_device=args.out_cam))
    except KeyboardInterrupt:
        print("\n[*] Interrompido pelo usuário.")
