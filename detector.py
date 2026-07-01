import os
import argparse
import sys
from pathlib import Path

try:
    import yt_dlp
    import whisper
    import cv2
    import pytesseract
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"⚠️ Erreur d'import : {e}")
    print("Les dépendances seront installées automatiquement par le workflow.")
    sys.exit(1)

# =====================================================================
# SCRIPT D'EXTRACTION DE CONTENU - APPRENTISSAGE
# =====================================================================
# Ce script extrait :
# 1. L'audio (MP3)
# 2. Le texte (OCR)
# 3. Les images clés (screenshots)
# D'une vidéo YouTube ou autre source
# =====================================================================

parser = argparse.ArgumentParser(description="Extracteur de contenu vidéo")
parser.add_argument('--url', required=True, help='URL de la vidéo (YouTube, etc.)')
parser.add_argument('--output', default='output', help='Dossier de sortie')
args = parser.parse_args()

# Créer le dossier de sortie
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

print(f"\n{'='*60}")
print(f" 🎬 EXTRACTEUR DE CONTENU - Mode Apprentissage")
print(f"{'='*60}")
print(f"URL : {args.url}")
print(f"Output : {output_dir}")
print(f"{'='*60}\n")

# =====================================================================
# ÉTAPE 1 : TÉLÉCHARGER LA VIDÉO
# =====================================================================
print("📥 [1/4] Téléchargement de la vidéo...")

video_file = output_dir / "video.mp4"

# Configuration pour contourner la détection de bot YouTube
ydl_opts = {
    'format': 'best[ext=mp4]',
    'outtmpl': str(output_dir / 'video'),
    'quiet': False,
    'no_warnings': False,
    # Techniques pour éviter la détection de bot
    'extractor_args': {
        'youtube': {
            'player_skip': ['js', 'configs'],
        }
    },
    'socket_timeout': 30,
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"  Téléchargement en cours...")
        info = ydl.extract_info(args.url, download=True)
        print(f"  ✅ Vidéo téléchargée\n")
except Exception as e:
    print(f"  ⚠️ Erreur téléchargement (essai alternatif) : {str(e)[:100]}")
    print(f"  Tentative avec méthode alternative...\n")
    
    # Essai alternatif avec options réduites
    ydl_opts_alt = {
        'format': 'best',
        'outtmpl': str(output_dir / 'video'),
        'quiet': False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_alt) as ydl:
            info = ydl.extract_info(args.url, download=True)
            print(f"  ✅ Vidéo téléchargée (méthode alternative)\n")
    except Exception as e2:
        print(f"  ❌ Impossible de télécharger : {e2}")
        print(f"  💡 Conseil : YouTube peut nécessiter une authentification.")
        print(f"  Essaie avec une URL différente ou une plateforme alternative.\n")
        sys.exit(1)

# =====================================================================
# ÉTAPE 2 : EXTRAIRE L'AUDIO ET TRANSCRIRE
# =====================================================================
print("🎙️ [2/4] Extraction de l'audio et transcription...")

audio_file = output_dir / "audio.mp3"
try:
    # Extraire l'audio avec FFmpeg (via yt-dlp)
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / 'audio'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }
    
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        print(f"  Extraction audio...")
        ydl.download([args.url])
    
    # Transcrire avec Whisper
    print(f"  Initialisation de Whisper (modèle 'tiny' - rapide)...")
    model = whisper.load_model("tiny")
    
    print(f"  Transcription en cours...")
    result = model.transcribe(str(audio_file))
    
    transcript_file = output_dir / "transcript.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    print(f"  ✅ Audio transcrit\n")
    print(f"📄 Texte extrait :")
    print(f"  {result['text'][:200]}...\n")
    
except Exception as e:
    print(f"  ⚠️ Erreur transcription : {e}\n")

# =====================================================================
# ÉTAPE 3 : EXTRACTION D'IMAGES ET OCR
# =====================================================================
print("📸 [3/4] Extraction de screenshots et OCR...")

try:
    # Chercher le fichier vidéo téléchargé
    video_files = list(output_dir.glob("video.*"))
    if not video_files:
        print(f"  ⚠️ Fichier vidéo introuvable\n")
        sys.exit(1)
    
    video_path = video_files[0]
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count == 0:
        print(f"  ⚠️ Impossible de lire la vidéo\n")
        cap.release()
        sys.exit(1)
    
    # Extraire 5 frames clés
    frame_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
    
    ocr_results = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Sauvegarder l'image
            img_file = output_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(img_file), frame)
            
            # OCR avec Tesseract
            try:
                text = pytesseract.image_to_string(frame)
                if text.strip():
                    ocr_results.append({
                        'frame': i,
                        'time': frame_idx / fps if fps > 0 else 0,
                        'text': text
                    })
                    print(f"  Frame {i} - Texte détecté : {text[:50]}...")
            except Exception as ocr_err:
                print(f"  ⚠️ OCR frame {i} : {ocr_err}")
    
    cap.release()
    
    # Sauvegarder les résultats OCR
    if ocr_results:
        ocr_file = output_dir / "ocr_results.txt"
        with open(ocr_file, 'w', encoding='utf-8') as f:
            for result in ocr_results:
                f.write(f"Frame {result['frame']} (t={result['time']:.2f}s):\n")
                f.write(f"{result['text']}\n")
                f.write("-" * 40 + "\n")
        print(f"  ✅ OCR complété\n")
    else:
        print(f"  ℹ️ Aucun texte détecté sur les frames\n")
        
except Exception as e:
    print(f"  ⚠️ Erreur extraction images : {e}\n")

# =====================================================================
# ÉTAPE 4 : RÉSUMÉ ET FICHIERS GÉNÉRÉS
# =====================================================================
print("✨ [4/4] Récapitulatif des fichiers générés...\n")

files_generated = list(output_dir.glob("*"))
print(f"📁 Fichiers dans '{output_dir}' :")
for f in files_generated:
    if f.is_file():
        size = f.stat().st_size
        size_str = f"{size / (1024*1024):.1f}MB" if size > 1024*1024 else f"{size / 1024:.1f}KB"
        print(f"   ✅ {f.name} ({size_str})")
    else:
        print(f"   📂 {f.name}/")

print(f"\n{'='*60}")
print(f" ✅ EXTRACTION COMPLÉTÉE")
print(f"{'='*60}")
print(f"\n📝 Fichiers disponibles :")
print(f"   - transcript.txt : Transcription audio")
print(f"   - ocr_results.txt : Texte extrait des images")
print(f"   - frame_0.jpg à frame_4.jpg : Screenshots")
print(f"   - audio.mp3 : Fichier audio")
print(f"   - video.* : Fichier vidéo original")
print(f"\n{'='*60}\n")
