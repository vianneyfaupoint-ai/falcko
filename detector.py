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
    from urllib.parse import urlencode
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"⚠️ Erreur d'import : {e}")
    print("Les dépendances seront installées automatiquement par le workflow.")
    sys.exit(1)

# =====================================================================
# SCRIPT D'EXTRACTION DE CONTENU - APPRENTISSAGE
# =====================================================================
# Ce script peut :
# 1. Chercher du contenu (Google Search)
# 2. Extraire l'audio (MP3)
# 3. Extraire le texte (OCR)
# 4. Extraire les images clés (screenshots)
# =====================================================================

parser = argparse.ArgumentParser(description="Extracteur de contenu vidéo multi-plateforme")
parser.add_argument('--search', help='Rechercher du contenu (ex: "falcko error 404")')
parser.add_argument('--url', help='URL directe de la vidéo (YouTube, Vimeo, etc.)')
parser.add_argument('--output', default='output', help='Dossier de sortie')
args = parser.parse_args()

# Créer le dossier de sortie
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

print(f"\n{'='*60}")
print(f" 🎬 EXTRACTEUR DE CONTENU - Mode Apprentissage")
print(f"{'='*60}\n")

# =====================================================================
# MODE 1 : RECHERCHE GOOGLE
# =====================================================================
if args.search:
    print(f"🔍 Recherche : '{args.search}'\n")
    
    # Utiliser DuckDuckGo au lieu de Google (moins bloquant)
    try:
        search_url = f"https://duckduckgo.com/html/?q={urlencode({'q': args.search})}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        print("  📡 Recherche en cours...")
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Chercher les liens vidéo
            results = []
            links = soup.find_all('a', {'class': 'result__url'})
            
            for link in links[:10]:  # Top 10 résultats
                try:
                    href = link.get('href')
                    text = link.get_text()
                    
                    if href and text:
                        results.append({
                            'title': text[:60],
                            'url': href
                        })
                except:
                    continue
            
            if results:
                print(f"\n  ✅ {len(results)} résultats trouvés :\n")
                for i, result in enumerate(results[:5], 1):
                    print(f"  {i}. {result['title']}")
                    print(f"     🔗 {result['url']}\n")
                
                # Sauvegarder les résultats
                results_file = output_dir / "search_results.txt"
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write(f"Recherche : {args.search}\n")
                    f.write("="*60 + "\n\n")
                    for i, result in enumerate(results, 1):
                        f.write(f"{i}. {result['title']}\n")
                        f.write(f"   URL: {result['url']}\n\n")
                
                print(f"  📄 Résultats sauvegardés dans 'search_results.txt'")
                print(f"  💡 Utilise --url pour télécharger une de ces vidéos\n")
            else:
                print(f"  ❌ Aucun résultat trouvé\n")
        else:
            print(f"  ⚠️ Erreur recherche (code {response.status_code})\n")
            
    except Exception as e:
        print(f"  ⚠️ Erreur recherche : {str(e)[:100]}\n")
    
    sys.exit(0)

# =====================================================================
# MODE 2 : EXTRACTION D'UNE URL
# =====================================================================
if not args.url:
    print("❌ Utilisation :")
    print("   Pour chercher : python detector.py --search 'votre recherche'")
    print("   Pour extraire : python detector.py --url 'https://...'")
    sys.exit(1)

print(f"URL : {args.url}")
print(f"Output : {output_dir}")
print(f"{'='*60}\n")

# =====================================================================
# ÉTAPE 1 : TÉLÉCHARGER LA VIDÉO
# =====================================================================
print("📥 [1/4] Téléchargement de la vidéo...")

video_file = output_dir / "video.mp4"

# Configuration pour contourner la détection de bot
ydl_opts = {
    'format': 'best[ext=mp4]',
    'outtmpl': str(output_dir / 'video'),
    'quiet': False,
    'no_warnings': False,
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
    
    # Essai alternatif
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
    
    frame_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
    ocr_results = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            img_file = output_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(img_file), frame)
            
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

print(f"\n{'='*60}")
print(f" ✅ EXTRACTION COMPLÉTÉE")
print(f"{'='*60}\n")
