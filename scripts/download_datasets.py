"""
Script pour télécharger les datasets d'évaluation RUPTA

Ce script télécharge automatiquement :
1. DB-Bio (Celebrity Biographies) 
2. PersonalReddit (Synthetic Reddit Comments)

depuis Google Drive et les extrait dans les bons répertoires.
"""

import os
import sys
import json
import gdown


def download_dbbio():
    """Télécharge le dataset DB-Bio"""
    print("\n" + "=" * 60)
    print("Téléchargement de DB-Bio")
    print("=" * 60)
    
    output_dir = "Dataset/evaluation/DB-Bio"
    os.makedirs(output_dir, exist_ok=True)
    
    # URL Google Drive du dataset DB-Bio
    url = "https://drive.google.com/uc?id=1oXWI2mh_mkrs2bZs4riGgbYbQoA9RNzD"
    output_file = os.path.join(output_dir, "db-bio.tar.gz")
    
    print(f"Téléchargement depuis Google Drive...")
    print(f"Destination : {output_file}")
    
    try:
        gdown.download(url, output_file, quiet=False)
        print(f"✅ Téléchargement terminé : {output_file}")
        
        # Extraction
        print(f"Extraction de l'archive...")
        import tarfile
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall(output_dir)
        print(f"✅ Extraction terminée dans {output_dir}")
        
        # Vérification
        files = os.listdir(output_dir)
        print(f"Fichiers extraits : {files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return False


def download_personalreddit():
    """Télécharge le dataset PersonalReddit"""
    print("\n" + "=" * 60)
    print("Téléchargement de PersonalReddit")
    print("=" * 60)
    
    output_dir = "Dataset/evaluation/PersonalReddit"
    os.makedirs(output_dir, exist_ok=True)
    
    # URL Google Drive du dataset PersonalReddit
    url = "https://drive.google.com/uc?id=1Z6Xs6zgsn7tkdcW5SElRzbSqUhZFLjwX"
    output_file = os.path.join(output_dir, "personalreddit.tar.gz")
    
    print(f"Téléchargement depuis Google Drive...")
    print(f"Destination : {output_file}")
    
    try:
        gdown.download(url, output_file, quiet=False)
        print(f"✅ Téléchargement terminé : {output_file}")
        
        # Extraction
        print(f"Extraction de l'archive...")
        import tarfile
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall(output_dir)
        print(f"✅ Extraction terminée dans {output_dir}")
        
        # Vérification
        files = os.listdir(output_dir)
        print(f"Fichiers extraits : {files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return False


def verify_datasets():
    """Vérifie que les datasets sont bien présents"""
    print("\n" + "=" * 60)
    print("Vérification des datasets")
    print("=" * 60)
    
    datasets = {
        "DB-Bio": "Dataset/evaluation/DB-Bio",
        "PersonalReddit": "Dataset/evaluation/PersonalReddit"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jsonl') or f.endswith('.json')]
            if files:
                print(f"✅ {name} : {len(files)} fichier(s) trouvé(s)")
                # Afficher un exemple
                for f in files[:3]:
                    file_path = os.path.join(path, f)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   - {f} ({size_mb:.2f} MB)")
            else:
                print(f"⚠️  {name} : répertoire présent mais aucun fichier .jsonl trouvé")
        else:
            print(f"❌ {name} : répertoire manquant")


def test_load_sample():
    """Teste le chargement d'exemples depuis les datasets"""
    print("\n" + "=" * 60)
    print("Test de chargement d'exemples")
    print("=" * 60)
    
    # Chercher les fichiers JSONL
    dbbio_files = []
    reddit_files = []
    
    dbbio_dir = "Dataset/evaluation/DB-Bio"
    reddit_dir = "Dataset/evaluation/PersonalReddit"
    
    if os.path.exists(dbbio_dir):
        dbbio_files = [os.path.join(dbbio_dir, f) for f in os.listdir(dbbio_dir) 
                      if f.endswith('.jsonl')]
    
    if os.path.exists(reddit_dir):
        reddit_files = [os.path.join(reddit_dir, f) for f in os.listdir(reddit_dir) 
                       if f.endswith('.jsonl')]
    
    # Test DB-Bio
    if dbbio_files:
        print(f"\n📄 Exemple DB-Bio ({dbbio_files[0]}):")
        try:
            with open(dbbio_files[0], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 1:  # Un seul exemple
                        break
                    data = json.loads(line)
                    print(f"  Text: {data.get('text', '')[:100]}...")
                    print(f"  People: {data.get('people', [])}")
                    print(f"  Label: {data.get('label', '')}")
        except Exception as e:
            print(f"  ❌ Erreur de lecture : {e}")
    
    # Test PersonalReddit
    if reddit_files:
        print(f"\n📄 Exemple PersonalReddit ({reddit_files[0]}):")
        try:
            with open(reddit_files[0], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 1:  # Un seul exemple
                        break
                    data = json.loads(line)
                    print(f"  Text: {data.get('text', '')[:100]}...")
                    print(f"  Attributes: {list(data.keys())}")
        except Exception as e:
            print(f"  ❌ Erreur de lecture : {e}")


def main():
    """Point d'entrée principal"""
    
    print("\n🚀 Téléchargement des datasets d'évaluation RUPTA")
    
    # Vérifier si gdown est installé
    try:
        import gdown
    except ImportError:
        print("\n⚠️  Le package 'gdown' n'est pas installé.")
        print("   Installation : pip install gdown")
        response = input("   Installer maintenant ? (y/n) : ")
        if response.lower() == 'y':
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✅ gdown installé")
        else:
            print("❌ Installation annulée")
            return
    
    # Menu
    print("\nOptions :")
    print("  1. Télécharger DB-Bio")
    print("  2. Télécharger PersonalReddit")
    print("  3. Télécharger les deux")
    print("  4. Vérifier les datasets")
    print("  5. Tester le chargement")
    
    choice = input("\nChoix (1-5) : ").strip()
    
    if choice == "1":
        download_dbbio()
    elif choice == "2":
        download_personalreddit()
    elif choice == "3":
        download_dbbio()
        download_personalreddit()
    elif choice == "4":
        verify_datasets()
    elif choice == "5":
        test_load_sample()
    else:
        print("Choix invalide")
        return
    
    # Vérification finale
    if choice in ["1", "2", "3"]:
        verify_datasets()
        test_load_sample()
    
    print("\n✅ Terminé !")


if __name__ == "__main__":
    main()
