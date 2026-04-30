#!/bin/bash
# =============================================================================
# setup_realtime.sh
# Configura os dispositivos v4l2loopback para o pipeline real-time.
#
# ATENÇÃO: Execute este script ANTES de iniciar o OBS.
# Ele carrega DOIS dispositivos em um único comando sudo:
#   /dev/video0 → OBS Virtual Camera
#   /dev/video2 → Sink manipulado (pipeline de ataque)
# =============================================================================

echo "============================================================"
echo "  Setup: v4l2loopback  →  /dev/video0 + /dev/video2"
echo "  ⚠  Execute ANTES de abrir o OBS Studio"
echo "============================================================"

# Pede a senha sudo UMA vez no início
sudo -v
if [ $? -ne 0 ]; then
    echo "❌ Falha na autenticação sudo. Abortando."
    exit 1
fi

# Remove instância anterior (se existir)
echo ""
echo "[1/3] Removendo módulo anterior (se carregado)..."
sudo modprobe -r v4l2loopback 2>/dev/null && echo "   Módulo removido." || echo "   Nenhum módulo carregado anteriormente."

# Carrega com DOIS devices em um único comando (evita expirar o sudo)
echo ""
echo "[2/3] Carregando v4l2loopback com video_nr=0,2..."
sudo modprobe v4l2loopback \
    video_nr=0,2 \
    card_label="OBSVirtualCamera,DeepfakeVirtual" \
    exclusive_caps=1

if [ $? -ne 0 ]; then
    echo "❌ Falha ao carregar v4l2loopback. Verifique se o pacote está instalado:"
    echo "   sudo apt install v4l2loopback-dkms v4l2loopback-utils"
    exit 1
fi

# Verifica devices criados
echo ""
echo "[3/3] Dispositivos v4l2 disponíveis:"
v4l2-ctl --list-devices

echo ""
echo "============================================================"
echo "✅  Devices criados:"
echo "    /dev/video0  →  OBS Virtual Camera (abra o OBS agora)"
echo "    /dev/video2  →  DeepfakeVirtual (sink do pipeline)"
echo "============================================================"
echo ""
echo "Passos seguintes:"
echo "  1. Abra o OBS Studio e clique em 'Iniciar câmera virtual'"
echo "     (ele usará /dev/video0 automaticamente)"
echo ""
echo "  2. Rode o pipeline de ataque:"
echo "     cd /mnt/BKP/Carol_Artigo/deepfaketeste"
echo "     python app/realtime_attack_pipeline.py \\"
echo "       --source 0 \\"
echo "       --sink   2 \\"
echo "       --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg"
echo "============================================================"
