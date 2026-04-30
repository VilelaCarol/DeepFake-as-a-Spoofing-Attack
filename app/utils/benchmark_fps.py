import os
import time
import numpy as np

try:
    import onnxruntime
except ImportError:
    print('ERRO: Execute este script dentro do container Docker.')
    exit(1)

model_path = '/app/facefusion/.assets/models/inswapper_128.onnx'

providers = [
    ('CUDAExecutionProvider', {
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'arena_extend_strategy': 'kSameAsRequested',
    }),
    'CPUExecutionProvider'
]

print('🔥 Configurando Otimização Máxima de Baixa Latência...')
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count()

session = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)

inputs = session.get_inputs()
target_name = inputs[0].name
emb_name = inputs[1].name

dummy_target = np.random.randn(1, 3, 128, 128).astype(np.float32)
dummy_embedding = np.random.randn(1, 512).astype(np.float32)

print('\n⏳ Executando WARMUP (Cache Otimizado)...')
for _ in range(50):
    session.run(None, {target_name: dummy_target, emb_name: dummy_embedding})

print('🚀 Rodando Benchmark Principal OTIMIZADO (1000 chamadas)...')
start_time = time.time()
num_iterations = 1000

for _ in range(num_iterations):
    session.run(None, {target_name: dummy_target, emb_name: dummy_embedding})

end_time = time.time()
total_time = end_time - start_time
fps = num_iterations / total_time
ms_per_frame = (total_time / num_iterations) * 1000

print('\n' + '='*60)
print(f' 📊 RESULTADOS DO BENCHMARK (ONNX OTIMIZADO) ')
print('='*60)
print(f' Tempo total gasto : {total_time:.2f} segundos')
print(f' ⏱️ Tempo por rosto: {ms_per_frame:.2f} milissegundos')
print(f' 🏎️ Taxa de Quadros: {fps:.2f} FPS')
print('='*60)
