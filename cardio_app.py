import uvicorn
import webbrowser
import threading
import time
import sys
import os
import scipy.stats.distributions          # ← añadido
import scipy.stats._distn_infrastructure 
import scipy.stats._continuous_distns
import scipy.stats._discrete_distns 

def start_server():
    """Inicia el servidor FastAPI."""
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    # Lanzar el servidor en un hilo separado para que no bloquee
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Dar tiempo a que el servidor arranque
    time.sleep(2)

    # Abrir el navegador por defecto
    webbrowser.open("http://127.0.0.1:8000")

    # Mantener vivo el proceso principal (si no, el hilo daemon se cierra)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Cerrando aplicación...")
        sys.exit(0)