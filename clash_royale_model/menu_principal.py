"""
SISTEMA DE RECOMENDACIÓN DE CARTAS CLASH ROYALE
Menú Principal
"""

import os
import sys
import subprocess


def clear_screen():
    """Limpia la pantalla"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Imprime el banner principal"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     CLASH ROYALE - SISTEMA DE RECOMENDACIÓN DE CARTAS    ║
    ║                      VERSIÓN 2.0                         ║
    ║                  (Dataset Actualizado 2026)              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Modelo de Deep Learning con arquitectura Attention + Dense
    Dataset: 33 cartas | Entrenamiento: 500 mazos
    """
    print(banner)


def print_menu():
    """Imprime el menú de opciones"""
    menu = """
    SELECCIONA UNA OPCIÓN:
    
    1. Ejecutar Modo Interactivo V3 (Recomendado)
       - Análisis estratégico avanzado
       - Foco en Ataque o Defensa
       - Recomendaciones variadas e inteligentes
    
    2. Ver Información del Proyecto
       - Descripción técnica
       - Arquitectura del modelo
       - Datos del dataset
    
    3. Salir
    
    """
    print(menu)


def show_info():
    """Muestra información del proyecto"""
    info = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           INFORMACIÓN DEL PROYECTO                        ║
    ╚═══════════════════════════════════════════════════════════╝
    
    OBJETIVO:
    Recomendador de cartas inteligente basado en Deep Learning
    que analiza tu mazo y sugiere la mejor carta para completarlo.
    
    DATASET:
    • Total de cartas: 33
    • Propiedades por carta:
      - Nombre, Tipo (Tropa, Hechizo, Estructura)
      - Costo de elixir, Daño, Vida
      - Foco de ataque, Tipo de ataque
      - Alcance, Duración
    
    ARQUITECTURA DEL MODELO:
    • Capa de embedding: 4D → 32D
    • Capa de attention multi-cabeza: 4 heads
    • Capas densas: 224 → 64 → 32 → 4
    • Función de pérdida: MSE (Mean Squared Error)
    • Optimizador: Adam (learning rate: 0.001)
    • Épocas de entrenamiento: 30
    
    DATOS DE ENTRENAMIENTO:
    • Mazos sintéticos generados: 500
    • División: 80% entrenamiento, 20% validación
    • Cada mazo: 7 cartas (input) + 1 carta objetivo (output)
    
    CARACTERÍSTICAS ESPECIALES:
    ✓ Análisis inteligente de necesidades del mazo
    ✓ Priorización automática según lo que falta
    ✓ Foco DEFENSA: Estructura + Hechizo + Tropa
    ✓ Foco ATAQUE: Tropas ofensivas y damage dealers
    ✓ Puntuaciones de efectividad (0-100%)
    ✓ Explicaciones dinámicas y contextualizadas
    
    TECNOLOGÍAS USADAS:
    • PyTorch 2.10.0 - Deep Learning
    • Pandas 3.0.0 - Procesamiento de datos
    • NumPy 2.4.2 - Computación numérica
    • Scikit-learn 1.8.0 - Normalización de datos
    • Openpyxl - Lectura de Excel
    
    CÓMO USAR:
    1. Ejecuta: python menu_principal.py
    2. Selecciona opción 1 (Modo Interactivo)
    3. Ingresa 7 cartas de tu mazo
    4. Elige entre ATAQUE (0) o DEFENSA (1)
    5. Recibe recomendaciones variadas e inteligentes
    
    ARCHIVO PRINCIPAL:
    • proyecto_v3_interactivo.py - Modo interactivo mejorado
    
    VERSIÓN: 2.0 (Mejorada con Foco en Defensa)
    Desarrollado con IA para el análisis estratégico de Clash Royale
    """
    print(info)
    input("\nPresiona Enter para volver al menú...")


def run_interactive_mode_v3():
    """Ejecuta el modo interactivo V3 mejorado"""
    print("\nIniciando proyecto_v3_interactivo.py...")
    print("Análisis estratégico avanzado con foco en Ataque/Defensa.\n")
    
    try:
        subprocess.run([sys.executable, "proyecto_v3_interactivo.py"], cwd=os.getcwd())
    except Exception as e:
        print(f"\nError: {e}")
        input("\nPresiona Enter para volver al menú...")


def main():
    """Función principal"""
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        try:
            opcion = input("Ingresa tu opcion (1-3): ").strip()
            
            if opcion == '1':
                clear_screen()
                run_interactive_mode_v3()
            
            elif opcion == '2':
                clear_screen()
                print_banner()
                show_info()
            
            elif opcion == '3':
                print("\n¡Gracias por usar el Sistema de Recomendación de Clash Royale!")
                print("¡Hasta pronto!\n")
                sys.exit(0)
            
            else:
                print("\nOpción no válida. Ingresa un número entre 1 y 3.")
                input("Presiona Enter para intentar de nuevo...")
        
        except KeyboardInterrupt:
            print("\n\n¡Adiós!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            input("Presiona Enter para intentar de nuevo...")


if __name__ == "__main__":
    main()
