# Proyecto Immobilier Scraper

## Instalación
1. Clonar el repositorio
2. Crear venv con Python 3.11: `py -3.11 -m venv venv`
3. Activar: `venv\Scripts\activate`
4. Instalar dependencias: `pip install -r requirements.txt`

## Estructura
- `src/scraper.py` - Web scraping
- `src/cleaner.py` - Limpieza de datos
- `src/analyzer.py` - Análisis estadístico
- `src/visualizer.py` - Visualizaciones
- `dashboard/app.py` - Streamlit dashboard
- `data/raw/` - Datos crudos
- `data/processed/` - Datos limpios